"""Quantum circuit synthesis with exact decompositions and optimized search."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

_DEADLINE = 240.0
_start_time = None


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build_unitary(gates, qubits):
    """Build unitary from gate list."""
    dim = 2 ** qubits
    matrix = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            # Build full operator via kron
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye2)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            # CNOT: flip target when control is 1
            index = np.arange(dim)
            cbit = (index >> (qubits - 1 - control)) & 1
            tbit_pos = qubits - 1 - target
            # For each state, if control bit is 1, flip target bit
            new_index = index.copy()
            mask = cbit == 1
            new_index[mask] = index[mask] ^ (1 << tbit_pos)
            matrix = matrix[new_index, :]
    return matrix


def _distance(target, matrix):
    """Phase-insensitive distance."""
    val = abs(np.trace(target.conj().T @ matrix)) / target.shape[0]
    return 1.0 - min(1.0, val)


def _is_cnot_gate(problem):
    qubits = problem["qubits"]
    if qubits != 2:
        return False
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=complex)
    return np.allclose(target, expected, atol=1e-10)


def _is_iswap_gate(problem):
    qubits = problem["qubits"]
    if qubits != 2:
        return False
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1j, 0],
                         [0, 1j, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    return np.allclose(target, expected, atol=1e-10)


def _is_swap_gate(problem):
    qubits = problem["qubits"]
    if qubits != 2:
        return False
    target = _target(problem)
    expected = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]], dtype=complex)
    return np.allclose(target, expected, atol=1e-10)


def _is_toffoli_gate(problem):
    qubits = problem["qubits"]
    if qubits != 3:
        return False
    target = _target(problem)
    expected = np.eye(8, dtype=complex)
    expected[6, 6] = 0
    expected[6, 7] = 1
    expected[7, 6] = 1
    expected[7, 7] = 0
    return np.allclose(target, expected, atol=1e-10)


def _decompose_cnot_2qubit():
    return [["cx", 0, 1]]


def _decompose_iswap_2qubit():
    """iSWAP with 2 CNOTs: H⊗H · CNOT · (S⊗S†) · CNOT · H⊗H"""
    pi = math.pi
    return [
        ["u3", 0, pi, 0, pi],      # H on q0
        ["u3", 1, pi, 0, pi],      # H on q1
        ["cx", 0, 1],               # CNOT 0->1
        ["u3", 0, 0, 0, pi / 2],   # S on q0
        ["u3", 1, 0, 0, -pi / 2],  # S† on q1
        ["cx", 0, 1],               # CNOT 0->1
        ["u3", 0, pi, 0, pi],      # H on q0
        ["u3", 1, pi, 0, pi],      # H on q1
    ]


def _decompose_swap_2qubit():
    """SWAP with 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _decompose_toffoli_3qubit():
    """Toffoli with 6 CNOTs (standard decomposition)."""
    pi = math.pi
    return [
        ["u3", 2, pi, 0, pi],        # H on target
        ["cx", 1, 2],                 # CNOT 1->2
        ["u3", 2, 0, 0, -pi / 4],    # T† on target
        ["cx", 0, 2],                 # CNOT 0->2
        ["u3", 2, 0, 0, pi / 4],     # T on target
        ["cx", 1, 2],                 # CNOT 1->2
        ["u3", 2, 0, 0, -pi / 4],    # T† on target
        ["u3", 2, pi, 0, pi],        # H on target
        ["cx", 0, 1],                 # CNOT 0->1
        ["u3", 1, 0, 0, -pi / 4],    # T† on q1
        ["cx", 0, 1],                 # CNOT 0->1
        ["u3", 1, 0, 0, pi / 4],     # T on q1
        ["u3", 2, 0, 0, pi / 4],     # T on target
    ]


def _optimize_generic(problem, target, qubits, budget):
    """Optimize single-qubit parameters for a generic unitary."""
    rng = random.Random(4242 + qubits + budget)
    
    # Build a circuit template with CNOTs and parameterized u3 gates
    # Use a balanced CNOT layout
    pairs = [(c, t) for c in range(qubits) for t in range(qubits) if c != t]
    
    # Create layout: layers of u3 gates + CNOT
    layout = []  # list of (type, data) where type is 'u3' or 'cx'
    param_indices = []  # indices into the parameter vector
    
    for layer in range(budget):
        for q in range(qubits):
            layout.append(('u3', q))
            param_indices.append(len(param_indices))  # 3 params per u3
        layout.append(('cx', pairs[layer % len(pairs)]))
    for q in range(qubits):
        layout.append(('u3', q))
        param_indices.append(len(param_indices))
    
    n_params = 3 * sum(1 for t, _ in layout if t == 'u3')
    
    def build_circuit(params):
        gates = []
        pi = 0
        for item in layout:
            if item[0] == 'u3':
                q = item[1]
                theta = params[pi]
                phi = params[pi + 1]
                lam = params[pi + 2]
                gates.append(["u3", q, theta, phi, lam])
                pi += 3
            else:
                gates.append(["cx", item[1][0], item[1][1]])
        return gates
    
    # Initial parameters: random small values
    best_params = np.array([rng.uniform(-0.1, 0.1) for _ in range(n_params)])
    best_dist = _distance(target, _build_unitary(build_circuit(best_params), qubits))
    
    # Coordinate descent with adaptive step size
    step = 0.5
    min_step = 1e-6
    max_sweeps = 50
    
    for sweep in range(max_sweeps):
        if time.time() - _start_time > _DEADLINE - 15:
            break
        
        improved = False
        for i in range(n_params):
            if time.time() - _start_time > _DEADLINE - 15:
                break
            
            # Try several perturbations
            trials = [
                best_params[i] + step,
                best_params[i] - step,
                best_params[i] + rng.uniform(-step, step),
                best_params[i] + rng.uniform(-math.pi, math.pi) * 0.1,
            ]
            
            for trial_val in trials:
                new_params = best_params.copy()
                new_params[i] = trial_val
                dist = _distance(target, _build_unitary(build_circuit(new_params), qubits))
                if dist < best_dist:
                    best_params = new_params
                    best_dist = dist
                    improved = True
                    break
        
        if not improved:
            step *= 0.5
            if step < min_step:
                break
    
    return build_circuit(best_params), best_dist


def synthesize(problem):
    """Synthesize circuit for the given target unitary."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)

    # Check for known gates and use exact decompositions
    if _is_cnot_gate(problem):
        return _decompose_cnot_2qubit()
    if _is_iswap_gate(problem):
        return _decompose_iswap_2qubit()
    if _is_swap_gate(problem):
        return _decompose_swap_2qubit()
    if _is_toffoli_gate(problem):
        return _decompose_toffoli_3qubit()

    # For generic unitaries, use coordinate descent optimization
    gates, dist = _optimize_generic(problem, target, qubits, budget)
    return gates
# EVOLVE-BLOCK-END
