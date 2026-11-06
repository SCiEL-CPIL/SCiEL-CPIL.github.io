"""Rewritten program: exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_DEADLINE = 255.0
_START_TIME = time.time()


def _check_deadline():
    return (time.time() - _START_TIME) > _DEADLINE


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _build(gates, qubits):
    """Build unitary from gate list using in-place row operations."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            # Apply u3 to qubit q: for each pair of rows differing only in bit q
            bit_mask = 1 << (qubits - 1 - q)
            for i in range(n):
                if i & bit_mask:
                    continue
                j = i | bit_mask
                # Rows i and j: apply 2x2 rotation
                # New row i = c * old_i - exp(i*lam)*s * old_j
                # New row j = exp(i*phi)*s * old_i + exp(i*(phi+lam))*c * old_j
                old_i = matrix[i, :].copy()
                old_j = matrix[j, :].copy()
                matrix[i, :] = c * old_i - np.exp(1j * lam) * s * old_j
                matrix[j, :] = np.exp(1j * phi) * s * old_i + np.exp(1j * (phi + lam)) * c * old_j
        else:
            _, control, target = gate
            c_mask = 1 << (qubits - 1 - control)
            t_mask = 1 << (qubits - 1 - target)
            for i in range(n):
                if i & c_mask:
                    j = i ^ t_mask
                    matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _apply_u3_inplace(matrix, q, theta, phi, lam, qubits):
    """Apply u3 gate to matrix in-place (left multiply)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    bit_mask = 1 << (qubits - 1 - q)
    for i in range(2 ** qubits):
        if i & bit_mask:
            continue
        j = i | bit_mask
        old_i = matrix[i, :].copy()
        old_j = matrix[j, :].copy()
        matrix[i, :] = c * old_i - np.exp(1j * lam) * s * old_j
        matrix[j, :] = np.exp(1j * phi) * s * old_i + np.exp(1j * (phi + lam)) * c * old_j


def _apply_cx_inplace(matrix, control, target, qubits):
    """Apply CNOT gate to matrix in-place (left multiply via row swap)."""
    c_mask = 1 << (qubits - 1 - control)
    t_mask = 1 << (qubits - 1 - target)
    for i in range(2 ** qubits):
        if i & c_mask:
            j = i ^ t_mask
            matrix[i], matrix[j] = matrix[j].copy(), matrix[i].copy()


def _circuit_matrix(gates, qubits):
    """Build unitary from gate list using in-place operations."""
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            _apply_u3_inplace(matrix, q, theta, phi, lam, qubits)
        else:
            _, control, target = gate
            _apply_cx_inplace(matrix, control, target, qubits)
    return matrix


def _decompose_2qubit_kak(U):
    """
    KAK decomposition of a 2-qubit unitary.
    Returns single-qubit gates and CNOT structure.
    U = (A ⊗ B) CNOT (C ⊗ D) CNOT (E ⊗ F) CNOT (G ⊗ H) * phase
    """
    # Use a simpler approach: direct coordinate descent with good initialization
    # For 2-qubit, we use the standard 3-CNOT decomposition
    # The circuit layout is:
    # q0: U0a --x-- U1a --x-- U2a --x-- U3a
    # q1: U0b ---o--- U1b ---o--- U2b ---o--- U3b
    # That's 8 single-qubit gates (24 params) + 3 CNOTs
    
    # Actually, let's use a different approach:
    # We'll use the fact that any 2-qubit unitary can be decomposed as:
    # U = (A ⊗ B) * CNOT * (C ⊗ D) * CNOT * (E ⊗ F) * CNOT * (G ⊗ H)
    # where A,B,C,D,E,F,G,H are single-qubit unitaries
    
    # For now, use coordinate descent with the in-place builder
    
    return None  # Placeholder


def _synthesize_2qubit(target, budget, qubits=2):
    """Synthesize a 2-qubit gate within CNOT budget using coordinate descent."""
    if budget < 1:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Build layout: alternating u3 and cx
    pairs = [(0, 1), (1, 0)]
    layout = []
    for layer in range(min(budget, 3)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
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
    
    def objective(values):
        if _check_deadline():
            return 1.0
        gates = circuit(values)
        matrix = _circuit_matrix(gates, qubits)
        return _distance(target, matrix)
    
    # Start from zero
    best = [0.0] * n_params
    best_dist = objective(best)
    
    # Coordinate descent
    rng = random.Random(4242 + qubits)
    
    max_sweeps = 30
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = math.pi / (4 * (sweep + 1))
        improved = False
        for i in range(n_params):
            if _check_deadline():
                break
            current_val = best[i]
            # Try multiple candidates
            candidates = [
                current_val + step,
                current_val - step,
                current_val + step/2,
                current_val - step/2,
                rng.uniform(-math.pi, math.pi),
            ]
            for trial in candidates:
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_dist:
                    best, best_dist = values, value
                    improved = True
        if not improved and sweep > 10:
            break
    
    return circuit(best)


def _synthesize_3qubit(target, budget, qubits=3):
    """Synthesize a 3-qubit gate within CNOT budget using coordinate descent."""
    if budget < 1:
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # Build layout: alternating u3 and cx with various CNOT pairs
    pairs = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]
    layout = []
    for layer in range(min(budget, 14)):
        layout += [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
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
    
    def objective(values):
        if _check_deadline():
            return 1.0
        gates = circuit(values)
        matrix = _circuit_matrix(gates, qubits)
        return _distance(target, matrix)
    
    # Start from zero
    best = [0.0] * n_params
    best_dist = objective(best)
    
    # Coordinate descent
    rng = random.Random(4242 + qubits)
    
    max_sweeps = 20
    for sweep in range(max_sweeps):
        if _check_deadline():
            break
        step = math.pi / (4 * (sweep + 1))
        improved = False
        for i in range(n_params):
            if _check_deadline():
                break
            current_val = best[i]
            candidates = [
                current_val + step,
                current_val - step,
                current_val + step/2,
                current_val - step/2,
                rng.uniform(-math.pi, math.pi),
            ]
            for trial in candidates:
                values = best[:]
                values[i] = trial
                value = objective(values)
                if value < best_dist:
                    best, best_dist = values, value
                    improved = True
        if not improved and sweep > 8:
            break
    
    return circuit(best)


def _known_circuit_cnot():
    return [["cx", 0, 1]]


def _known_circuit_iswap():
    """iSWAP = (H ⊗ H) CNOT (S† ⊗ S) CNOT (H ⊗ H)"""
    h = [math.pi/2, 0.0, math.pi]
    s = [0.0, 0.0, math.pi/2]
    sdag = [0.0, 0.0, -math.pi/2]
    return [
        ["u3", 0, h[0], h[1], h[2]],
        ["u3", 1, h[0], h[1], h[2]],
        ["cx", 0, 1],
        ["u3", 0, sdag[0], sdag[1], sdag[2]],
        ["u3", 1, s[0], s[1], s[2]],
        ["cx", 0, 1],
        ["u3", 0, h[0], h[1], h[2]],
        ["u3", 1, h[0], h[1], h[2]],
    ]


def _known_circuit_swap():
    """SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)"""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _known_circuit_toffoli():
    """Toffoli(0,1,2) decomposition with 6 CNOTs."""
    return [
        ["u3", 2, 0.0, 0.0, -math.pi/4],
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, math.pi/4],
        ["cx", 0, 2],
        ["u3", 2, 0.0, 0.0, -math.pi/4],
        ["cx", 1, 2],
        ["u3", 2, 0.0, 0.0, math.pi/4],
        ["u3", 1, math.pi/2, 0.0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 1, 0.0, 0.0, -math.pi/4],
        ["cx", 2, 1],
        ["u3", 1, 0.0, 0.0, math.pi/4],
        ["cx", 0, 1],
        ["u3", 1, math.pi/2, 0.0, -math.pi/2],
    ]


def _is_close_to_reference(target, ref, tol=1e-9):
    """Check if target is close to reference up to global phase."""
    # Remove global phase
    n = target.shape[0]
    # Find a non-zero element
    idx = np.unravel_index(np.argmax(np.abs(target)), target.shape)
    if np.abs(target[idx]) < 1e-15:
        return False
    phase = target[idx] / np.abs(target[idx])
    target_phase_removed = target / phase
    return np.allclose(target_phase_removed, ref, atol=tol)


def synthesize(problem):
    """
    Synthesize a quantum circuit for the given target unitary within the CNOT budget.
    """
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check deadline
    if _check_deadline():
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    # For 2-qubit gates
    if qubits == 2:
        # Check known gates
        cnot_ref = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        if _is_close_to_reference(target, cnot_ref):
            return _known_circuit_cnot()
        
        iswap_ref = np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex)
        if _is_close_to_reference(target, iswap_ref):
            return _known_circuit_iswap()
        
        swap_ref = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        if _is_close_to_reference(target, swap_ref):
            return _known_circuit_swap()
        
        # Generic 2-qubit: use coordinate descent
        return _synthesize_2qubit(target, budget, qubits)
    
    # For 3-qubit gates
    elif qubits == 3:
        # Check if target is Toffoli
        toffoli_ref = np.zeros((8, 8), dtype=complex)
        for i in range(8):
            c0 = (i >> 2) & 1
            c1 = (i >> 1) & 1
            j = i
            if c0 == 1 and c1 == 1:
                j = i ^ 1
            toffoli_ref[j, i] = 1.0
        if _is_close_to_reference(target, toffoli_ref):
            return _known_circuit_toffoli()
        
        # Generic 3-qubit: use coordinate descent
        return _synthesize_3qubit(target, budget, qubits)
    
    # Fallback
    return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]


# EVOLVE-BLOCK-END
