"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

import numpy as np

# Safety margin before the 270s wall-clock kill
_DEADLINE = 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    eye = np.eye(2, dtype=complex)
    matrix = np.eye(2 ** qubits, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            single = np.array([[c, -np.exp(1j * lam) * s],
                               [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]])
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            index = np.arange(2 ** qubits)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_close(a, b, tol=1e-5):
    """Check if two matrices are close up to global phase."""
    # Compute overlap
    tr = np.trace(a.conj().T @ b)
    if np.abs(tr) < 1e-9:
        return False
    phase = tr / np.abs(tr)
    # Check if a is close to b * phase
    return np.allclose(a, b * phase, atol=tol)

def _exact_cnot_01():
    return [['cx', 0, 1]]

def _exact_cnot_10():
    return [['cx', 1, 0]]

def _exact_iswap():
    # iSWAP = 
    # 0 0 0 0
    # 0 0 i 0
    # 0 i 0 0
    # 0 0 0 0
    # Standard decomposition:
    # u3(pi/2, 0, pi/2) on 0
    # u3(pi/2, pi/2, 0) on 0
    # cx(0, 1)
    # u3(pi/2, 0, pi/2) on 1
    # u3(pi/2, pi/2, 0) on 1
    # cx(0, 1)
    # u3(pi/2, pi/2, 0) on 0
    # u3(pi/2, 0, pi/2) on 1
    return [
        ['u3', 0, math.pi/2, 0, math.pi/2],
        ['u3', 0, math.pi/2, math.pi/2, 0],
        ['cx', 0, 1],
        ['u3', 1, math.pi/2, 0, math.pi/2],
        ['u3', 1, math.pi/2, math.pi/2, 0],
        ['cx', 0, 1],
        ['u3', 0, math.pi/2, math.pi/2, 0],
        ['u3', 1, math.pi/2, 0, math.pi/2]
    ]

def _exact_swap():
    return [
        ['cx', 0, 1],
        ['cx', 1, 0],
        ['cx', 0, 1]
    ]

def _exact_toffoli():
    # Toffoli (CCNOT) on q0, q1, q2 (0,1 are controls, 2 is target)
    # Standard 4-CNOT decomposition
    return [
        ['u3', 2, math.pi/2, 0, math.pi/2],
        ['cx', 1, 2],
        ['u3', 2, math.pi/4, 0, 0],
        ['cx', 0, 2],
        ['u3', 2, -math.pi/4, 0, 0],
        ['cx', 1, 2],
        ['u3', 2, math.pi/4, 0, 0],
        ['cx', 0, 2],
        ['u3', 2, -math.pi/4, 0, 0],
        ['u3', 2, math.pi/2, math.pi/2, 0],
        ['u3', 0, math.pi/2, math.pi/2, 0],
        ['u3', 1, math.pi/2, math.pi/2, 0]
    ]

def _detect_and_exact(problem):
    qubits = problem["qubits"]
    target = _target(problem)
    
    # Check 2-qubit gates
    if qubits == 2:
        # CNOT 0->1
        cnot_01 = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1], [0,0,1,0]], dtype=complex)
        if _is_close(target, cnot_01):
            return _exact_cnot_01()
            
        # CNOT 1->0
        cnot_10 = np.array([[1,0,0,0], [0,0,0,1], [0,0,1,0], [0,1,0,0]], dtype=complex)
        if _is_close(target, cnot_10):
            return _exact_cnot_10()
            
        # iSWAP
        iswap = np.array([[0,0,0,0], [0,0,1j,0], [0,1j,0,0], [0,0,0,0]], dtype=complex)
        if _is_close(target, iswap):
            return _exact_iswap()
            
        # SWAP
        swap = np.array([[1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]], dtype=complex)
        if _is_close(target, swap):
            return _exact_swap()
            
    # Check 3-qubit gates
    if qubits == 3:
        # Toffoli (0,1 -> 2)
        toffoli = np.zeros((8,8), dtype=complex)
        for i in range(8):
            toffoli[i,i] = 1
        toffoli[6,6] = 0
        toffoli[7,7] = 0
        toffoli[6,7] = 1
        toffoli[7,6] = 1
        if _is_close(target, toffoli):
            return _exact_toffoli()
            
        # Other Toffoli permutations?
        # The task says "the Toffoli gate", implying a specific one or any.
        # Usually q0,q1 control q2.
        # If it's a different permutation, we might need to check.
        # But "the Toffoli" usually refers to the standard CCNOT.
        
    return None

def synthesize(problem):
    """
    Synthesize a quantum circuit.
    1. Try exact decompositions for known gates.
    2. Fallback to coordinate descent on a CNOT ladder.
    """
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    exact = _detect_and_exact(problem)
    if exact is not None:
        return exact
        
    # Define the CNOT layout: a "ladder" structure
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else: # qubits == 3
        pairs = [(0, 1), (1, 2), (2, 0)]
        
    layout = []
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    for i in range(budget):
        layout.append(["cx", *pairs[i % len(pairs)]])
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
            
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
        
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    n_params = 3 * num_u3
    
    param_map = []
    k = 0
    for i, gate in enumerate(layout):
        if gate[0] == "u3":
            param_map.append((i, 0))
            param_map.append((i, 1))
            param_map.append((i, 2))
            k += 3

    def get_circuit(params):
        gates = [list(g) for g in layout]
        for p_idx, (gate_idx, angle_idx) in enumerate(param_map):
            gates[gate_idx][2 + angle_idx] = params[p_idx]
        return gates

    def objective(params):
        if _check_deadline():
            return 1.0
        gates = get_circuit(params)
        matrix = _build(gates, qubits)
        return _distance(target, matrix)

    best_params = [0.0] * n_params
    best_dist = objective(best_params)
    
    rng = random.Random(4242 + qubits + budget)
    max_sweeps = 20
    
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
            
        step = 1.0 / (1 + sweep)
        improved = False
        
        for i in range(n_params):
            if _check_deadline():
                break
                
            current_val = best_params[i]
            
            candidates = [
                current_val + step,
                current_val - step,
                rng.uniform(-math.pi, math.pi)
            ]
            
            for trial in candidates:
                trial = (trial + math.pi) % (2 * math.pi) - math.pi
                
                new_params = best_params[:]
                new_params[i] = trial
                
                val = objective(new_params)
                
                if val < best_dist:
                    best_params = new_params
                    best_dist = val
                    improved = True
                    break
                    
        if not improved and sweep > 5:
            if step < 1e-4:
                break

    return get_circuit(best_params)
# EVOLVE-BLOCK-END
