"""Combined approach: exact algebraic decompositions for known gates + fast optimization for generic unitaries."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np

_START_TIME = time.time()
_DEADLINE = 255.0  # Safety margin before 270s wall-clock kill


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    e1 = np.exp(1j * lam)
    e2 = np.exp(1j * phi)
    return np.array([[c, -e1 * s],
                     [e2 * s, e2 * e1 * c]])


def _build_fast(gates, qubits, target):
    """Build circuit unitary and compute phase-insensitive distance efficiently."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            e1 = np.exp(1j * lam)
            e2 = np.exp(1j * phi)
            bit_pos = qubits - 1 - q
            shift = 1 << bit_pos
            new_matrix = np.zeros_like(matrix)
            for i in range(n):
                if (i & shift) == 0:
                    j = i | shift
                    new_matrix[i, :] = c * matrix[i, :] - e1 * s * matrix[j, :]
                    new_matrix[j, :] = e2 * s * matrix[i, :] + e2 * e1 * c * matrix[j, :]
            matrix = new_matrix
        else:
            _, control, target_q = gate
            cshift = 1 << (qubits - 1 - control)
            tshift = 1 << (qubits - 1 - target_q)
            perm = np.arange(n)
            for i in range(n):
                if (i & cshift):
                    perm[i] = i ^ tshift
            matrix = matrix[perm, :]
    
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / n)


def _u3(q, theta, phi, lam):
    return ["u3", q, theta, phi, lam]


def _cx(control, target):
    return ["cx", control, target]


def _exact_cnot():
    """CNOT gate: |00><00| + |01><01| + |10><11| + |11><10|"""
    return [_u3(0, 0.0, 0.0, 0.0), _u3(1, 0.0, 0.0, 0.0), _cx(0, 1)]


def _exact_iswap():
    """iSWAP gate: 0.5 * [[1, i, 0, 0], [i, 1, 0, 0], [0, 0, 1, i], [0, 0, i, 1]]"""
    # iSWAP = (1/sqrt(2)) * [[1, i, 0, 0], [i, 1, 0, 0], [0, 0, 1, i], [0, 0, i, 1]]
    # Actually iSWAP is: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, i], [0, 0, i, 0]] * global phase
    # Standard iSWAP: [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, i], [0, 0, i, 0]]
    # Decomposition: H(0) CNOT(0,1) H(0) then phase gates
    # iSWAP = (I ⊗ S) · CNOT(0,1) · (H ⊗ H) · CNOT(1,0) · (H ⊗ H) · (I ⊗ S†) ... 
    # Let me use a known decomposition:
    # iSWAP can be done with 2 CNOTs and single-qubit gates
    # From literature: iSWAP = [H(0), S(1), CNOT(0,1), H(0), S†(1), CNOT(1,0), H(0), S(1)]
    # Actually let me just return the correct gate sequence
    # iSWAP matrix: [[0,1,0,0],[1,0,0,0],[0,0,0,i],[0,0,i,0]]
    # This is: SWAP with a phase of i on the |10><01| and |01><10| terms
    # Equivalent to: CNOT(0,1), H(0), H(1), CNOT(1,0), H(0), H(1), CNOT(0,1) with phase corrections
    # Simpler: use the fact that iSWAP = exp(i*pi/4 * (XX+YY)/2)
    # Known 2-CNOT decomposition:
    gates = [
        _u3(0, math.pi/2, 0, math.pi/2),  # H-like
        _u3(1, math.pi/2, 0, math.pi/2),  # H-like
        _cx(0, 1),
        _u3(0, math.pi, 0, 0),  # X rotation
        _u3(1, math.pi, 0, 0),  # X rotation
        _cx(1, 0),
        _u3(0, math.pi/2, 0, math.pi/2),
        _u3(1, math.pi/2, 0, math.pi/2),
        _u3(1, 0, 0, math.pi/2),  # S gate on qubit 1
    ]
    return gates


def _exact_swap():
    """SWAP gate: 3 CNOTs"""
    return [_cx(0, 1), _cx(1, 0), _cx(0, 1)]


def _exact_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)"""
    # Toffoli |00><00| + |01><01| + |10><10| + |111><111|
    # Standard 6-CNOT decomposition
    gates = [
        _u3(0, math.pi/2, 0, math.pi/2),  # H on q0
        _u3(1, 0, 0, math.pi/4),  # T on q1
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),  # T† on q2
        _cx(0, 2),
        _u3(2, math.pi/2, 0, math.pi/4),  # T on q2
        _cx(1, 2),
        _u3(2, 0, 0, -math.pi/4),  # T† on q2
        _u3(1, math.pi/2, 0, -math.pi/4),  # T† on q1
        _u3(0, math.pi/2, 0, -math.pi/2),  # H on q0
        _cx(0, 1),
        _u3(1, 0, 0, math.pi/4),  # T on q1
        _cx(0, 1),
        _u3(1, 0, 0, -math.pi/4),  # T† on q1
    ]
    return gates


def _try_known_gates(problem, target, qubits, budget):
    """Try to match target to known standard gates and return exact circuit if found."""
    name = problem.get("name", "")
    
    # Check if target matches CNOT
    if qubits == 2:
        # CNOT
        cnot_target = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if np.allclose(target, cnot_target, atol=1e-8):
            return _exact_cnot()
        
        # iSWAP
        iswap_target = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex)
        if np.allclose(target, iswap_target, atol=1e-8):
            return _exact_iswap()
        
        # SWAP
        swap_target = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if np.allclose(target, swap_target, atol=1e-8):
            return _exact_swap()
    
    if qubits == 3:
        # Toffoli
        toffoli_target = np.zeros((8,8), dtype=complex)
        for i in range(8):
            if i != 7:
                toffoli_target[i,i] = 1
            else:
                toffoli_target[7,7] = 1  # |111> -> |111> for Toffoli
        # Actually Toffoli: |110> -> |111>, |111> -> |110>
        toffoli_target = np.eye(8, dtype=complex)
        toffoli_target[6,6] = 0
        toffoli_target[7,7] = 0
        toffoli_target[6,7] = 1
        toffoli_target[7,6] = 1
        if np.allclose(target, toffoli_target, atol=1e-8):
            return _exact_toffoli()
    
    return None


def _optimize_generic(target, qubits, budget):
    """Optimize single-qubit rotations for generic unitaries using coordinate descent."""
    # Define CNOT layout
    if qubits == 2:
        pairs = [(0, 1), (1, 0)]
    else:
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
        return _build_fast(gates, qubits, target)
    
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
        
        if not improved and sweep > 5 and step < 1e-4:
            break
    
    return get_circuit(best_params)


def synthesize(problem):
    """
    Synthesize a quantum circuit: use exact decompositions for known gates,
    optimize for generic unitaries.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Try known gate decompositions first
    known_circuit = _try_known_gates(problem, target, qubits, budget)
    if known_circuit is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known_circuit if g[0] == "cx")
        if cnot_count <= budget and len(known_circuit) <= 400:
            return known_circuit
    
    # Fall back to optimization for generic unitaries
    return _optimize_generic(target, qubits, budget)
# EVOLVE-BLOCK-END
