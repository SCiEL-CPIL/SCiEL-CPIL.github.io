"""Search-based synthesis with deadline guard and exact decompositions for known gates."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 250.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_fast(gates, qubits):
    """Build unitary from gate list using precomputed basis."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s], [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            # Build full operator via kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else np.eye(2, dtype=complex))
            matrix = op @ matrix
        else:
            _, control, target = gate
            idx = np.arange(dim)
            bit = (idx >> (qubits - 1 - control)) & 1
            perm = idx ^ (bit << (qubits - 1 - target))
            matrix = matrix[perm, :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_circuit_close(target, gates, qubits, tol=1e-6):
    """Check if circuit matches target within tolerance."""
    U = _build_fast(gates, qubits)
    return _distance(target, U) < tol


def _decompose_cnot(qubits):
    """Exact CNOT gate on qubit 0 (control) -> qubit 1 (target) for 2 qubits."""
    if qubits == 2:
        return [
            ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
        ]
    return None


def _decompose_iswap(qubits):
    """iSWAP gate for 2 qubits."""
    if qubits == 2:
        return [
            ["u3", 0, math.pi / 2, 0.0, math.pi / 2],
            ["u3", 1, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["u3", 0, math.pi / 2, -math.pi / 2, 0.0],
            ["u3", 1, math.pi / 2, -math.pi / 2, 0.0],
        ]
    return None


def _decompose_swap(qubits):
    """SWAP gate for 2 qubits."""
    if qubits == 2:
        return [
            ["cx", 0, 1],
            ["cx", 1, 0],
            ["cx", 0, 1],
        ]
    return None


def _decompose_toffoli(qubits):
    """Toffoli gate for 3 qubits: controls 0,1 target 2."""
    if qubits == 3:
        return [
            ["u3", 2, math.pi / 2, 0.0, math.pi / 2],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, -math.pi / 2, 0.0],
            ["cx", 0, 2],
            ["u3", 2, math.pi / 2, math.pi / 2, 0.0],
            ["cx", 1, 2],
            ["u3", 2, math.pi / 2, -math.pi / 2, 0.0],
            ["u3", 1, 0.0, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 1, math.pi / 4, 0.0, 0.0],
            ["cx", 0, 1],
            ["u3", 1, -math.pi / 4, 0.0, math.pi / 2],
            ["cx", 0, 1],
            ["u3", 1, 0.0, 0.0, math.pi / 2],
        ]
    return None


def _try_exact_decomposition(name, target, qubits, budget):
    """Try known exact decompositions."""
    gates = None
    if name == "cnot" and qubits == 2:
        gates = _decompose_cnot(qubits)
    elif name == "iswap" and qubits == 2:
        gates = _decompose_iswap(qubits)
    elif name == "swap" and qubits == 2:
        gates = _decompose_swap(qubits)
    elif name == "toffoli" and qubits == 3:
        gates = _decompose_toffoli(qubits)
    
    if gates is not None:
        cnot_count = sum(1 for g in gates if g[0] == "cx")
        if cnot_count <= budget and len(gates) <= 400:
            if _is_circuit_close(target, gates, qubits, tol=1e-8):
                return gates
    return None


def _coordinate_descent(target, layout, qubits, budget, rng, time_limit):
    """Coordinate descent to optimize single-qubit rotations."""
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = [0.0] * n_params
    best_dist = _distance(target, _build_fast(circuit(best), qubits))
    
    start = time.time()
    max_sweeps = 50
    for sweep in range(max_sweeps):
        if time.time() - start > time_limit:
            break
        step = 2.0 / (1 + sweep)
        for i in range(n_params):
            if time.time() - start > time_limit:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                val = _distance(target, _build_fast(circuit(values), qubits))
                if val < best_dist:
                    best, best_dist = values, val
    return circuit(best), best_dist


def synthesize(problem):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try exact decomposition first
    exact = _try_exact_decomposition(name, target, qubits, budget)
    if exact is not None:
        return exact
    
    # Build layout with CNOTs
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits + hash(name) % 1000)
    
    layout = []
    for layer in range(budget):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
        layout.append(["cx", *pairs[layer % len(pairs)]])
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Time remaining for this problem
    elapsed = time.time() - _start_time
    time_left = _DEADLINE - elapsed
    if time_left < 5:
        # Return trivial circuit
        return [g if g[0] == "cx" else ["u3", g[1], 0.0, 0.0, 0.0] for g in layout]
    
    # Allocate time: at least 10s per problem, cap at 30s
    time_alloc = min(30.0, max(10.0, time_left / 7.0))
    
    gates, dist = _coordinate_descent(target, layout, qubits, budget, rng, time_alloc)
    
    # Verify CNOT count
    cnot_count = sum(1 for g in gates if g[0] == "cx")
    if cnot_count > budget:
        # Truncate
        gates = [g for g in gates if g[0] != "cx"][:budget] + [g for g in gates if g[0] == "cx"]
    
    return gates
# EVOLVE-BLOCK-END
