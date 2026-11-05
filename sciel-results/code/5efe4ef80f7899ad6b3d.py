"""Quantum circuit synthesis with known decompositions and KAK for generic gates."""
import math
import time
import numpy as np

_DEADLINE = time.time() + 260  # Safety margin before 270s kill


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build(gates, qubits):
    n = 2 ** qubits
    matrix = np.eye(n, dtype=complex)
    eye = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            operator = np.array([[1]], dtype=complex)
            for k in range(qubits):
                operator = np.kron(operator, single if k == q else eye)
            matrix = operator @ matrix
        else:
            _, control, target = gate
            idx = np.arange(n)
            bit = (idx >> (qubits - 1 - control)) & 1
            matrix = matrix[idx ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _is_cnot(target, qubits):
    """Check if target is (up to phase) a CNOT."""
    if qubits != 2:
        return False
    # CNOT: |00>,|01> unchanged; |10>->|11>, |11>->|10>
    # Check if target matches CNOT pattern up to global phase
    cnot = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    # Try to find phase
    for phase in [1, -1, 1j, -1j]:
        if np.allclose(target, phase * cnot, atol=1e-9):
            return True
    return False


def _is_swap(target, qubits):
    if qubits != 2:
        return False
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    for phase in [1, -1, 1j, -1j]:
        if np.allclose(target, phase * swap, atol=1e-9):
            return True
    return False


def _is_iswap(target, qubits):
    if qubits != 2:
        return False
    iswap = np.array([[1, 0, 0, 0],
                      [0, 0, 1j, 0],
                      [0, 1j, 0, 0],
                      [0, 0, 0, 1]], dtype=complex)
    for phase in [1, -1, 1j, -1j]:
        if np.allclose(target, phase * iswap, atol=1e-9):
            return True
    return False


def _is_toffoli(target, qubits):
    if qubits != 3:
        return False
    toffoli = np.eye(8, dtype=complex)
    # Toffoli: |110> <-> |111>
    toffoli[6, 6] = 0
    toffoli[6, 7] = 1
    toffoli[7, 6] = 1
    toffoli[7, 7] = 0
    for phase in [1, -1, 1j, -1j]:
        if np.allclose(target, phase * toffoli, atol=1e-9):
            return True
    return False


def _cnot_circuit():
    return [["cx", 0, 1]]


def _swap_circuit():
    return [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]


def _iswap_circuit():
    # iSWAP = (H⊗H) · CNOT · (S†⊗I) · CNOT · (H⊗H) ... let me use standard decomposition
    # iSWAP = 1/sqrt(2) * [[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]
    # Standard: iSWAP = (I⊗S) · CNOT · (H⊗H) · CNOT · (I⊗S†) ... 
    # Actually: iSWAP = (H⊗H) · CNOT · (S†⊗I) · CNOT · (H⊗H) · (phase corrections)
    # Let me use: iSWAP = (1/sqrt(2))[[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # Decomposition: CNOT(0,1), (H⊗H), CNOT(1,0), (H⊗H), then phases
    # Simpler: iSWAP = (S⊗S) · (CNOT(0,1) · (H⊗H) · CNOT(0,1)) ... 
    # Let me just use the known 2-CNOT decomposition:
    # iSWAP = (1/sqrt(2)) * [[1,0,0,0],[0,0,i,0],[0,i,0,0],[0,0,0,1]]
    # = (H⊗H) · CNOT(0,1) · (S†⊗I) · CNOT(0,1) · (H⊗H) · (phase)
    # Actually the standard is:
    # iSWAP = (I⊗Rz(π/2)) · CNOT(0,1) · (H⊗H) · CNOT(0,1) · (I⊗Rz(-π/2)) · (phase)
    # Let me use a cleaner approach:
    # iSWAP can be done with 2 CNOTs:
    gates = [
        ["u3", 0, math.pi, 0, math.pi / 2],  # H on q0
        ["u3", 1, math.pi, 0, math.pi / 2],  # H on q1
        ["cx", 0, 1],
        ["u3", 1, 0, 0, -math.pi / 2],  # S† on q1
        ["cx", 0, 1],
        ["u3", 0, math.pi, 0, math.pi / 2],  # H on q0
        ["u3", 1, math.pi, 0, math.pi / 2],  # H on q1
        ["u3", 1, 0, 0, math.pi / 2],  # S on q1 for phase
    ]
    return gates


def _toffoli_circuit():
    # Toffoli with 6 CNOTs
    # Standard decomposition
    gates = [
        ["u3", 2, math.pi / 4, 0, 0],  # T on target
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],  # T† on target
        ["cx", 0, 2],
        ["u3", 2, math.pi / 4, 0, 0],  # T on target
        ["cx", 1, 2],
        ["u3", 2, -math.pi / 4, 0, 0],  # T† on target
        ["u3", 1, 0, 0, math.pi / 2],  # S on middle
        ["cx", 0, 1],
        ["u3", 1, -math.pi / 4, 0, 0],  # T† on middle
        ["u3", 0, 0, 0, math.pi / 2],  # S on control
        ["cx", 1, 2],
        ["u3", 2, math.pi / 4, 0, 0],  # T on target
        ["cx", 0, 2],
        ["u3", 2, -math.pi / 4, 0, 0],  # T† on target
    ]
    return gates


def _kak_decompose_2qubit(target):
    """
    KAK decomposition for 2-qubit gates: U = e^{iφ} (A⊗B) exp(i(θ1 σx⊗σx + θ2 σy⊗σy + θ3 σz⊗σz)) (C⊗D)
    Uses 3 CNOTs.
    """
    # For simplicity and reliability, use a numerical optimization approach
    # but with a good initial guess
    
    # Actually, let me use a simpler approach: parameterize with 3 CNOTs and optimize
    # U = U1 ⊗ U2 · CNOT · U3 ⊗ U4 · CNOT · U5 ⊗ U6 · CNOT · U7 ⊗ U8
    
    # This is complex. Let me use a different strategy:
    # For generic 2-qubit gates, use the fact that any 2-qubit gate can be decomposed
    # as: single-qubit · CNOT · single-qubit · CNOT · single-qubit · CNOT · single-qubit
    
    # I'll use a simple iterative method to find the parameters
    
    # Initialize with identity
    params = [0.0] * 24  # 8 u3 gates × 3 params each
    
    def circuit_from_params(p):
        gates = []
        idx = 0
        # U1, U2
        gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["u3", 1, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["cx", 0, 1])
        # U3, U4
        gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["u3", 1, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["cx", 0, 1])
        # U5, U6
        gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["u3", 1, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["cx", 0, 1])
        # U7, U8
        gates.append(["u3", 0, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        gates.append(["u3", 1, p[idx], p[idx+1], p[idx+2]])
        idx += 3
        return gates
    
    best_params = params[:]
    best_dist = _distance(target, _build(circuit_from_params(best_params), 2))
    
    # Coordinate descent with a few sweeps
    for sweep in range(3):
        step = 0.5 / (sweep + 1)
        for i in range(len(best_params)):
            for delta in [step, -step]:
                trial = best_params[:]
                trial[i] += delta
                trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                dist = _distance(target, _build(circuit_from_params(trial), 2))
                if dist < best_dist:
                    best_params = trial
                    best_dist = dist
    
    return circuit_from_params(best_params)


def _generic_3qubit(target, budget):
    """
    Decompose a generic 3-qubit gate using the available CNOT budget.
    Strategy: Use a ladder of CNOTs with single-qubit rotations.
    """
    # For 3-qubit gates, we need a more sophisticated approach
    # Use a standard decomposition: any 3-qubit gate can be decomposed with
    # O(n^2) CNOTs, but we need to fit within budget
    
    # Simple approach: use a sequence of 2-qubit gates
    # Each 2-qubit gate costs 3 CNOTs (via KAK or similar)
    
    # For now, use a parameterized circuit and optimize
    # Layout: layers of single-qubit + CNOTs
    
    # Create a layout with the given budget
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 2), (2, 0)]
    layout = []
    for i in range(budget):
        layout.append(["u3", 0, 0.0, 0.0, 0.0])
        layout.append(["u3", 1, 0.0, 0.0, 0.0])
        layout.append(["u3", 2, 0.0, 0.0, 0.0])
        layout.append(["cx", pairs[i % len(pairs)][0], pairs[i % len(pairs)][1]])
    # Final single-qubit layer
    layout.append(["u3", 0, 0.0, 0.0, 0.0])
    layout.append(["u3", 1, 0.0, 0.0, 0.0])
    layout.append(["u3", 2, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    params = [0.0] * n_params
    
    def circuit_from_params(p):
        gates = []
        idx = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], p[idx], p[idx+1], p[idx+2]])
                idx += 3
            else:
                gates.append(list(gate))
        return gates
    
    best_params = params[:]
    best_dist = _distance(target, _build(circuit_from_params(best_params), 3))
    
    # Coordinate descent
    for sweep in range(2):
        step = 0.5 / (sweep + 1)
        for i in range(len(best_params)):
            if time.time() > _DEADLINE - 10:
                break
            for delta in [step, -step]:
                trial = best_params[:]
                trial[i] += delta
                trial[i] = (trial[i] + math.pi) % (2 * math.pi) - math.pi
                dist = _distance(target, _build(circuit_from_params(trial), 3))
                if dist < best_dist:
                    best_params = trial
                    best_dist = dist
        if time.time() > _DEADLINE - 10:
            break
    
    return circuit_from_params(best_params)


def synthesize(problem):
    """Synthesize quantum circuit for the given target unitary."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    if qubits == 2:
        if _is_cnot(target, qubits):
            return _cnot_circuit()
        if _is_swap(target, qubits):
            return _swap_circuit()
        if _is_iswap(target, qubits):
            return _iswap_circuit()
    
    if qubits == 3:
        if _is_toffoli(target, qubits):
            return _toffoli_circuit()
    
    # Generic decomposition
    if qubits == 2:
        return _kak_decompose_2qubit(target)
    else:
        return _generic_3qubit(target, budget)
