"""Quantum circuit synthesis: exact known gates + fast coordinate descent for generics."""
import math
import random
import time

import numpy as np

_DEADLINE = time.time() + 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    """Build the unitary matrix from a gate list using in-place row swaps for CNOT."""
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                op = np.kron(op, single if k == q else eye2)
            U = op @ U
        else:
            _, c, t = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - c)) & 1
            U = U[idx ^ (bit << (qubits - 1 - t)), :]
    return U


def _distance(target, U):
    dim = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / dim)


# --- Exact gate helpers ---

def _h(q):
    return ['u3', q, math.pi / 2, 0, math.pi]


def _s(q):
    return ['u3', q, 0, 0, math.pi / 2]


def _sdag(q):
    return ['u3', q, 0, 0, -math.pi / 2]


def _t(q):
    return ['u3', q, math.pi / 4, 0, 0]


def _tdag(q):
    return ['u3', q, -math.pi / 4, 0, 0]


def _cnot(c, t):
    return ['cx', c, t]


def _decomp_cnot_01():
    return [_cnot(0, 1)]


def _decomp_cnot_10():
    return [_cnot(1, 0)]


def _decomp_swap():
    return [_cnot(0, 1), _cnot(1, 0), _cnot(0, 1)]


def _decomp_iswap():
    """iSWAP via (H⊗H) · CX(0,1) · (S†⊗S) · CX(0,1) · (H⊗H)."""
    return [_h(0), _h(1), _cnot(0, 1), _sdag(0), _s(1), _cnot(0, 1), _h(0), _h(1)]


def _decomp_toffoli(c1, c2, t):
    """6-CNOT Toffoli decomposition for controls c1, c2 and target t."""
    gates = []
    gates.append(_h(t))
    gates.append(_cnot(c1, t))
    gates.append(_tdag(t))
    gates.append(_t(c2))
    gates.append(_cnot(c2, t))
    gates.append(_t(t))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, t))
    gates.append(_t(c2))
    gates.append(_tdag(t))
    gates.append(_h(t))
    gates.append(_cnot(c1, c2))
    gates.append(_t(c1))
    gates.append(_tdag(c2))
    gates.append(_cnot(c1, c2))
    return gates


def _match_known(target, qubits):
    """Try to match the target against known gate decompositions."""
    if qubits == 2:
        candidates = [
            _decomp_cnot_01(),
            _decomp_cnot_10(),
            _decomp_swap(),
            _decomp_iswap(),
        ]
        for gates in candidates:
            U = _build(gates, qubits)
            if _distance(target, U) < 1e-10:
                return gates
    elif qubits == 3:
        for c1 in range(3):
            for c2 in range(3):
                if c2 == c1:
                    continue
                t = 3 - c1 - c2
                gates = _decomp_toffoli(c1, c2, t)
                U = _build(gates, qubits)
                if _distance(target, U) < 1e-10:
                    return gates
    return None


def _generic_synthesis(target, qubits, budget):
    """Coordinate descent synthesis for generic unitaries with a ladder CNOT layout."""
    n = 2 ** qubits
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]

    # Build layout: interleaved u3 gates and CNOTs
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[layer % len(pairs)][0], pairs[layer % len(pairs)][1]])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])

    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    best = [0.0] * num_params

    def circuit(values):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k + 1], values[k + 2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates

    best_U = _build(circuit(best), qubits)
    best_dist = _distance(target, best_U)

    # Coordinate descent with random restarts for the first few sweeps
    max_sweeps = 35
    rng = random.Random(12345 + qubits * 7 + budget)

    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = 1.0 / (1.0 + sweep * 0.3)
        improved = False
        for i in range(num_params):
            if _check_deadline():
                break
            base = best[i]
            trials = [base + step, base - step, base + step * 0.5, base - step * 0.5]
            if sweep < 8:
                trials.append(base + rng.uniform(-math.pi, math.pi))
            for trial in trials:
                values = best[:]
                values[i] = trial
                U = _build(circuit(values), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U
                    improved = True
                    break
        if not improved and sweep > 10:
            break

    # Final polish with smaller steps
    for polish in range(15):
        if _check_deadline():
            break
        step = 0.05 / (1.0 + polish)
        for i in range(num_params):
            if _check_deadline():
                break
            base = best[i]
            for trial in (base + step, base - step):
                values = best[:]
                values[i] = trial
                U = _build(circuit(values), qubits)
                d = _distance(target, U)
                if d < best_dist:
                    best = values
                    best_dist = d
                    best_U = U

    return circuit(best)


def synthesize(problem):
    """
    Synthesize a quantum circuit.
    1. Try exact decompositions for known gates.
    2. Fallback to coordinate descent on a CNOT ladder.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Try known gate matching first
    known = _match_known(target, qubits)
    if known is not None:
        cnot_count = sum(1 for g in known if g[0] == 'cx')
        if cnot_count <= budget:
            return known

    # Generic synthesis
    return _generic_synthesis(target, qubits, budget)
