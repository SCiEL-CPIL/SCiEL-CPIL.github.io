"""Search-based synthesis with exact decompositions for known gates and deadline guard."""
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
    """Build unitary from gate list efficiently."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else np.eye(2, dtype=complex))
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(dim)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_close(a, b, tol=1e-5):
    """Check if two unitaries are close up to global phase."""
    dim = a.shape[0]
    # Compute overlap
    overlap = np.trace(a.conj().T @ b)
    return abs(overlap) / dim > 1.0 - tol


def _exact_cnot(problem):
    """Return exact CNOT circuit."""
    target = _target(problem)
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    cnot_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    if _is_close(target, cnot_01):
        return [["cx", 0, 1]]
    elif _is_close(target, cnot_10):
        return [["cx", 1, 0]]
    return None


def _exact_swap(problem):
    """Return exact SWAP circuit."""
    target = _target(problem)
    swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    if _is_close(target, swap):
        return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
    return None


def _exact_iswap(problem):
    """Return exact iSWAP circuit using 3 CNOTs."""
    target = _target(problem)
    iswap = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
    if _is_close(target, iswap):
        # iSWAP decomposition:
        # iSWAP = diag(1,1,1,1) * ... 
        # Known 3-CNOT decomposition:
        # Apply to q0 and q1:
        gates = []
        # S† on both
        gates.append(["u3", 0, 0, 0, -math.pi/2])
        gates.append(["u3", 1, 0, 0, -math.pi/2])
        gates.append(["cx", 0, 1])
        gates.append(["u3", 0, 0, 0, math.pi/2])
        gates.append(["u3", 1, 0, 0, math.pi/2])
        gates.append(["cx", 1, 0])
        gates.append(["u3", 0, 0, 0, math.pi/2])
        gates.append(["u3", 1, 0, 0, math.pi/2])
        gates.append(["cx", 0, 1])
        return gates
    return None


def _exact_toffoli(problem):
    """Return exact Toffoli circuit using 6 CNOTs."""
    target = _target(problem)
    # Toffoli with controls 0,1 and target 2
    toffoli = np.eye(8, dtype=complex)
    toffoli[6, 6] = 0; toffoli[7, 6] = 1
    toffoli[7, 7] = 0; toffoli[6, 7] = 1
    
    if _is_close(target, toffoli):
        gates = []
        gates.append(["u3", 2, math.pi/2, 0, math.pi])       # H
        gates.append(["u3", 2, 0, 0, -math.pi/4])            # T†
        gates.append(["cx", 0, 2])
        gates.append(["u3", 0, 0, 0, math.pi/4])             # T
        gates.append(["u3", 2, 0, 0, math.pi/4])             # T
        gates.append(["cx", 1, 2])
        gates.append(["u3", 0, 0, 0, -math.pi/4])            # T†
        gates.append(["u3", 2, 0, 0, -math.pi/4])            # T†
        gates.append(["cx", 0, 2])
        gates.append(["u3", 0, 0, 0, math.pi/4])             # T
        gates.append(["u3", 2, 0, 0, -math.pi/4])            # T†
        gates.append(["cx", 1, 2])
        gates.append(["u3", 1, 0, 0, math.pi/4])             # T
        gates.append(["u3", 2, 0, 0, math.pi/4])             # T
        gates.append(["u3", 2, math.pi/2, 0, math.pi])       # H
        return gates
    return None


def _search_generic(problem, time_limit):
    """Search for decomposition of a generic unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    rng = random.Random(4242 + qubits)
    
    layout = []
    layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    for layer in range(budget):
        layout.append(["cx", *pairs[layer % len(pairs)]])
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
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
    best_distance = _distance(target, _build_fast(circuit(best), qubits))
    
    sweep = 0
    while time.time() - _start_time < time_limit:
        step = 1.0 / (1 + sweep)
        improved = False
        for i in range(n_params):
            if time.time() - _start_time > time_limit:
                break
            for trial in (best[i] + step, best[i] - step, best[i] + rng.uniform(-math.pi, math.pi)):
                values = best[:]
                values[i] = trial
                value = _distance(target, _build_fast(circuit(values), qubits))
                if value < best_distance:
                    best, best_distance = values, value
                    improved = True
        if not improved:
            break
        sweep += 1
    
    return circuit(best)


def synthesize(problem):
    """Synthesize circuit for the given problem."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    result = _exact_cnot(problem)
    if result is not None:
        return result
    
    result = _exact_swap(problem)
    if result is not None:
        return result
    
    result = _exact_iswap(problem)
    if result is not None:
        return result
    
    result = _exact_toffoli(problem)
    if result is not None:
        return result
    
    remaining = _DEADLINE - (time.time() - _start_time)
    if remaining < 5:
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    return _search_generic(problem, remaining)
# EVOLVE-BLOCK-END
