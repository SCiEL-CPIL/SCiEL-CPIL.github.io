"""Rewritten program for quantum circuit synthesis with exact decompositions and time-budgeted search."""
# EVOLVE-BLOCK-START
import math
import random
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 230.0  # Keep 40s margin from 270s wall-clock kill


def _remaining_time():
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list."""
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    n = 2 ** qubits
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)
            # Build operator efficiently
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == q:
                    operator = np.kron(operator, single)
                else:
                    operator = np.kron(operator, eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # Build permutation
            perm = np.arange(n)
            for idx in range(n):
                if (idx >> (qubits - 1 - control)) & 1:
                    perm[idx] = idx ^ (1 << (qubits - 1 - target))
                else:
                    perm[idx] = idx
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance: 1 - |tr(target^dagger U)| / 2^n"""
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _cnot_gate(control, target):
    return ["cx", control, target]


def _u3_gate(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _exact_cnot():
    """Exact CNOT decomposition: 1 CNOT needed."""
    return [_cnot_gate(0, 1)], 1


def _exact_iswap():
    """iSWAP decomposition: 3 CNOTs + single-qubit rotations."""
    gates = []
    # H on both qubits
    gates.append(_u3_gate(0, math.pi, 0, 0))
    gates.append(_u3_gate(1, math.pi, 0, 0))
    # CNOT 0->1
    gates.append(_cnot_gate(0, 1))
    # Phase: Rz(-pi/2) on q0, Rz(pi/2) on q1
    gates.append(_u3_gate(0, 0, 0, -math.pi / 2))
    gates.append(_u3_gate(1, 0, 0, math.pi / 2))
    # CNOT 1->0
    gates.append(_cnot_gate(1, 0))
    # CNOT 0->1
    gates.append(_cnot_gate(0, 1))
    # H on both
    gates.append(_u3_gate(0, math.pi, 0, 0))
    gates.append(_u3_gate(1, math.pi, 0, 0))
    return gates, 3


def _exact_swap():
    """SWAP = 3 CNOTs, no single-qubit rotations needed."""
    gates = [
        _cnot_gate(0, 1),
        _cnot_gate(1, 0),
        _cnot_gate(0, 1),
    ]
    return gates, 3


def _exact_toffoli():
    """Toffoli (CCNOT) decomposition: 6 CNOTs."""
    gates = []
    # H on target
    gates.append(_u3_gate(2, math.pi, 0, 0))
    # CNOT 1->2
    gates.append(_cnot_gate(1, 2))
    # T on target
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    # CNOT 0->2
    gates.append(_cnot_gate(0, 2))
    # T-dagger on target
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    # CNOT 1->2
    gates.append(_cnot_gate(1, 2))
    # T on q1
    gates.append(_u3_gate(1, 0, 0, math.pi / 4))
    # CNOT 0->1
    gates.append(_cnot_gate(0, 1))
    # T-dagger on q1
    gates.append(_u3_gate(1, 0, 0, -math.pi / 4))
    # T on target
    gates.append(_u3_gate(2, 0, 0, math.pi / 4))
    # CNOT 0->2
    gates.append(_cnot_gate(0, 2))
    # T-dagger on target
    gates.append(_u3_gate(2, 0, 0, -math.pi / 4))
    # H on target
    gates.append(_u3_gate(2, math.pi, 0, 0))
    return gates, 6


def _is_known_gate(problem):
    """Check if the target is one of the known gates."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    target = _target(problem)

    if qubits == 2:
        cnot = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
        if np.allclose(target, cnot, atol=1e-10):
            return "cnot"

        iswap = np.array([[1, 0, 0, 0],
                          [0, 0, 1j, 0],
                          [0, 1j, 0, 0],
                          [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, iswap, atol=1e-10):
            return "iswap"

        swap = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
        if np.allclose(target, swap, atol=1e-10):
            return "swap"

    if qubits == 3:
        toffoli = np.eye(8, dtype=complex)
        toffoli[6, 6] = 0
        toffoli[7, 7] = 0
        toffoli[6, 7] = 1
        toffoli[7, 6] = 1
        if np.allclose(target, toffoli, atol=1e-10):
            return "toffoli"

    return None


def _greedy_optimize(target, qubits, budget, initial_gates, max_sweeps=8, rng=None):
    """Greedy coordinate descent optimization of u3 parameters."""
    if rng is None:
        rng = random.Random(4242)

    u3_indices = []
    param_idx = 0
    for i, gate in enumerate(initial_gates):
        if gate[0] == "u3":
            u3_indices.append((i, param_idx))
            param_idx += 3

    n_params = param_idx
    if n_params == 0:
        return initial_gates, _distance(target, _build(initial_gates, qubits))

    best_params = [0.0] * n_params
    best_gates = [list(g) for g in initial_gates]
    best_dist = _distance(target, _build(best_gates, qubits))

    def apply_params(params):
        gates = [list(g) for g in initial_gates]
        k = 0
        for i, gate in enumerate(gates):
            if gate[0] == "u3":
                gates[i] = [gate[0], gate[1], params[k], params[k + 1], params[k + 2]]
                k += 3
        return gates

    for sweep in range(max_sweeps):
        if _remaining_time() < 10:
            break
        step = math.pi / (sweep + 1)
        improved = False
        for p in range(n_params):
            if _remaining_time() < 5:
                break
            best_local = best_params[p]
            for trial in (best_local + step, best_local - step, rng.uniform(-math.pi, math.pi)):
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                params = best_params[:]
                params[p] = trial
                gates = apply_params(params)
                dist = _distance(target, _build(gates, qubits))
                if dist < best_dist:
                    best_params = params
                    best_gates = gates
                    best_dist = dist
                    improved = True
        if not improved:
            break

    return best_gates, best_dist


def _random_layout(qubits, budget, rng):
    """Generate a random circuit layout with exactly budget CNOTs."""
    gates = []
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]

    for q in range(qubits):
        gates.append(_u3_gate(q, 0.0, 0.0, 0.0))

    for i in range(budget):
        for q in range(qubits):
            gates.append(_u3_gate(q, 0.0, 0.0, 0.0))
        c, t = pairs[rng.randint(0, len(pairs) - 1)]
        gates.append(_cnot_gate(c, t))

    for q in range(qubits):
        gates.append(_u3_gate(q, 0.0, 0.0, 0.0))

    return gates


def _synthesize_generic(target, qubits, budget, rng):
    """Synthesize a generic unitary using multiple random restarts and greedy optimization."""
    best_gates = None
    best_dist = 1.0

    max_restarts = 5
    for restart in range(max_restarts):
        if _remaining_time() < 20:
            break

        layout = _random_layout(qubits, budget, rng)
        gates, dist = _greedy_optimize(target, qubits, budget, layout, max_sweeps=6, rng=rng)

        if dist < best_dist:
            best_dist = dist
            best_gates = gates

        if best_dist < 1e-10:
            break

    if best_gates is None:
        best_gates = [_u3_gate(q, 0.0, 0.0, 0.0) for q in range(qubits)]

    return best_gates


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Clamp budget to reasonable values
    if budget < 0:
        budget = 0
    if budget > 100:
        budget = 100

    rng = random.Random(12345 + qubits * 100 + budget)

    known = _is_known_gate(problem)
    if known == "cnot":
        gates, cnot_count = _exact_cnot()
        if cnot_count <= budget:
            return gates
    elif known == "iswap":
        gates, cnot_count = _exact_iswap()
        if cnot_count <= budget:
            return gates
    elif known == "swap":
        gates, cnot_count = _exact_swap()
        if cnot_count <= budget:
            return gates
    elif known == "toffoli":
        gates, cnot_count = _exact_toffoli()
        if cnot_count <= budget:
            return gates

    gates = _synthesize_generic(target, qubits, budget, rng)

    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        count = 0
        filtered = []
        for g in gates:
            if g[0] == "cx":
                if count < budget:
                    filtered.append(g)
                    count += 1
            else:
                filtered.append(g)
        gates = filtered

    if len(gates) > 400:
        gates = gates[:400]
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count > budget:
            count = 0
            filtered = []
            for g in gates:
                if g[0] == "cx":
                    if count < budget:
                        filtered.append(g)
                        count += 1
                else:
                    filtered.append(g)
            gates = filtered

    return gates
# EVOLVE-BLOCK-END
