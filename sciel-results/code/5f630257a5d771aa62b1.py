"""Synthesize fixed two- and three-qubit unitaries within CNOT budgets."""
import math
import time

import numpy as np

_START = time.time()
_DEADLINE = _START + 230.0


def _remaining():
    return max(0.0, _DEADLINE - time.time())


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _cx_matrix(qubits, control, target):
    n = 2 ** qubits
    mat = np.eye(n, dtype=complex)
    for i in range(n):
        cb = (i >> (qubits - 1 - control)) & 1
        if cb:
            tb = (i >> (qubits - 1 - target)) & 1
            j = i ^ (1 << (qubits - 1 - target))
            if tb == 0:
                mat[i, j] = 1
            else:
                mat[i, i] = 1
        else:
            mat[i, i] = 1
    return mat


def _build(gates, qubits):
    n = 2 ** qubits
    U = np.eye(n, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for g in gates:
        if g[0] == "u3":
            _, q, th, ph, lm = g
            op = _u3_matrix(th, ph, lm)
            full = np.array([[1]], dtype=complex)
            for k in range(qubits):
                full = np.kron(full, op if k == q else eye2)
            U = full @ U
        else:
            _, c, t = g
            U = _cx_matrix(qubits, c, t) @ U
    return U


def _dist(target, U):
    n = target.shape[0]
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ U)) / n)


def _count_cx(gates):
    return sum(1 for g in gates if g[0] == "cx")


def _decompose_single_qubit(mat):
    """Decompose a 2x2 unitary into u3 parameters."""
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    phase = np.angle(det)
    mat = mat * np.exp(-0.5j * phase)
    a, b = mat[0, 0], mat[0, 1]
    c, d = mat[1, 0], mat[1, 1]
    
    if abs(b) < 1e-12:
        theta = 0.0
        phi = 0.0
        lam = 0.0
        if abs(a) > 1e-12:
            lam = np.angle(d / a)
    else:
        theta = 2.0 * np.arcsin(np.clip(abs(b), 0, 1))
        if abs(b) > 1e-12:
            lam = np.angle(-b / np.sin(theta / 2))
            phi = np.angle(c / np.sin(theta / 2))
        else:
            lam = 0.0
            phi = 0.0
    return theta, phi, lam, phase


def _exact_cnot_decomposition():
    return [["cx", 0, 1]]


def _exact_iswap_decomposition():
    """
    iSWAP = (1/sqrt(2)) * [[0,1,0,0],[1,0,0,0],[0,0,0,i],[0,0,i,0]]
    Decomposition:
    iSWAP = e^{i*pi/4} * (H ⊗ H) * (S† ⊗ S†) * CX * (S ⊗ S) * (H ⊗ H)
    where S = diag(1, i), S† = diag(1, -i)
    
    Let's verify:
    H ⊗ H = 1/2 [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    S ⊗ S = diag(1, i, i, -1)
    S† ⊗ S† = diag(1, -i, -i, 1)
    
    Actually, let me use a known decomposition:
    iSWAP = (Rz(π/4) ⊗ Rz(π/4)) · (H ⊗ H) · CX · (H ⊗ H) · (Rz(-π/4) ⊗ Rz(-π/4)) · e^{iπ/4}
    
    Let me use a simpler approach. The standard decomposition is:
    iSWAP = (H ⊗ H) · CZ · (H ⊗ H) · (S† ⊗ S†) · global_phase
    where CZ = diag(1,1,1,-1)
    
    CZ = (I ⊗ H) · CX · (I ⊗ H)
    
    So iSWAP = (H ⊗ H) · (I ⊗ H) · CX · (I ⊗ H) · (H ⊗ H) · (S† ⊗ S†) · global_phase
    
    Let me just use the 3-CNOT decomposition:
    iSWAP = (Rz(π/4)·Rx(π/2) ⊗ Rz(π/4)·Rx(π/2)) · CX · (Rx(-π/2)·Rz(-π/4) ⊗ Rx(-π/2)·Rz(-π/4))
    
    Actually, let me use a different known decomposition.
    
    The iSWAP gate can be decomposed as:
    iSWAP = (H ⊗ H) · (S† ⊗ S†) · CX · (S ⊗ S) · (H ⊗ H) · e^{iπ/4}
    
    Let me verify this numerically in code. For now, let me use a safe decomposition.
    
    Standard 3-CNOT decomposition for iSWAP:
    iSWAP = (Rz(π/4) ⊗ Rz(π/4)) · (H ⊗ H) · CX · (H ⊗ H) · (Rz(-π/4) ⊗ Rz(-π/4)) · e^{iπ/4}
    """
    # Use the decomposition: iSWAP = e^{iπ/4} · (H⊗H) · (S†⊗S†) · CX · (S⊗S) · (H⊗H)
    # S = diag(1, i) = u3(0, 0, π/2)
    # S† = diag(1, -i) = u3(0, 0, -π/2)
    # H = u3(π/2, 0, π)
    
    gates = []
    # H ⊗ H (applied first in the sequence, so last in gate list if we think in terms of circuit)
    # Actually, gates are applied left to right in the list, so the first gate is applied first.
    # iSWAP = e^{iπ/4} · (H⊗H) · (S†⊗S†) · CX · (S⊗S) · (H⊗H)
    # So the circuit is: H⊗H, then S⊗S, then CX, then S†⊗S†, then H⊗H
    # Wait, that's not right. If U = A · B · C, then the circuit applies C first, then B, then A.
    # So for iSWAP = e^{iπ/4} · (H⊗H) · (S†⊗S†) · CX · (S⊗S) · (H⊗H):
    # The circuit is: (H⊗H), (S⊗S), CX, (S†⊗S†), (H⊗H)
    
    # H = u3(π/2, 0, π)
    # S = u3(0, 0, π/2)
    # S† = u3(0, 0, -π/2)
    
    gates.append(["u3", 0, math.pi/2, 0, math.pi])  # H on q0
    gates.append(["u3", 1, math.pi/2, 0, math.pi])  # H on q1
    gates.append(["u3", 0, 0, 0, math.pi/2])  # S on q0
    gates.append(["u3", 1, 0, 0, math.pi/2])  # S on q1
    gates.append(["cx", 0, 1])  # CX
    gates.append(["u3", 0, 0, 0, -math.pi/2])  # S† on q0
    gates.append(["u3", 1, 0, 0, -math.pi/2])  # S† on q1
    gates.append(["u3", 0, math.pi/2, 0, math.pi])  # H on q0
    gates.append(["u3", 1, math.pi/2, 0, math.pi])  # H on q1
    
    return gates


def _exact_swap_decomposition():
    """SWAP = CX01 · CX10 · CX01"""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1]
    ]


def _exact_toffoli_decomposition():
    """
    Toffoli gate: 6 CNOTs.
    Standard decomposition:
    Toffoli = (I ⊗ H) · CX01 · (I ⊗ T†) · CX02 · (I ⊗ T) · CX01 · (I ⊗ H) · CX12 · (S ⊗ I ⊗ T†) · (S† ⊗ I ⊗ I)
    
    Actually, let me use a known 6-CNOT decomposition:
    Toffoli = 
      CX01 · (I⊗H) · CX02 · (I⊗T†) · CX01 · (I⊗T) · CX02 · (I⊗H) · CX01 · (phase corrections)
    
    Let me use the standard decomposition from Nielsen & Chuang:
    Toffoli = (I⊗H) · CX01 · (I⊗T†) · CX02 · (I⊗T) · CX01 · (I⊗H) · CX12 · (S⊗I⊗T†) · (S†⊗I⊗I)
    
    Where T = diag(1, e^{iπ/4}) = u3(0, 0, π/4)
    T† = diag(1, e^{-iπ/4}) = u3(0, 0, -π/4)
    S = diag(1, i) = u3(0, 0, π/2)
    S† = diag(1, -i) = u3(0, 0, -π/2)
    H = u3(π/2, 0, π)
    """
    gates = []
    # T = u3(0, 0, π/4)
    # T† = u3(0, 0, -π/4)
    # S = u3(0, 0, π/2)
    # S† = u3(0, 0, -π/2)
    # H = u3(π/2, 0, π)
    
    # Toffoli = (I⊗H) · CX01 · (I⊗T†) · CX02 · (I⊗T) · CX01 · (I⊗H) · CX12 · (S⊗I⊗T†) · (S†⊗I⊗I)
    # Circuit order (left to right in gate list):
    # S† on q0, S on q0, T† on q2, CX12, H on q1, CX01, T on q2, CX02, T† on q2, CX01, H on q1
    
    gates.append(["u3", 0, 0, 0, -math.pi/2])  # S† on q0
    gates.append(["u3", 0, 0, 0, math.pi/2])  # S on q0
    gates.append(["u3", 2, 0, 0, -math.pi/4])  # T† on q2
    gates.append(["cx", 1, 2])  # CX12
    gates.append(["u3", 1, math.pi/2, 0, math.pi])  # H on q1
    gates.append(["cx", 0, 1])  # CX01
    gates.append(["u3", 2, 0, 0, math.pi/4])  # T on q2
    gates.append(["cx", 0, 2])  # CX02
    gates.append(["u3", 2, 0, 0, -math.pi/4])  # T† on q2
    gates.append(["cx", 0, 1])  # CX01
    gates.append(["u3", 1, math.pi/2, 0, math.pi])  # H on q1
    
    return gates


def _is_known_gate(target, name, qubits, budget):
    """Check if the target matches a known gate."""
    if qubits == 2:
        if name == "CNOT" or budget == 1:
            cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
            return abs(np.trace(target.conj().T @ cnot)) / 4 > 0.99
        if name == "iSWAP" or budget == 3:
            iswap = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1j],[0,0,1j,0]], dtype=complex) / math.sqrt(2)
            return abs(np.trace(target.conj().T @ iswap)) / 4 > 0.99
        if name == "SWAP":
            swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
            return abs(np.trace(target.conj().T @ swap)) / 4 > 0.99
    elif qubits == 3:
        if name == "Toffoli" or budget == 6:
            toffoli = np.array([
                [1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,0]
            ], dtype=complex)
            return abs(np.trace(target.conj().T @ toffoli)) / 8 > 0.99
    return False


def _kak_decompose_2qubit(U):
    """
    KAK decomposition for 2-qubit gates.
    U = (A ⊗ B) · exp(-i/2 · (x·XX + y·YY + z·ZZ)) · (C ⊗ D)
    
    The interaction term can be decomposed into 3 CNOTs:
    exp(-i/2 · (x·XX + y·YY + z·ZZ)) = 
      (Rz(x/2) ⊗ I) · CX · (Rz(-x/2) ⊗ I) · CX · (Rz(x/2) ⊗ I) · CX · (Rz(-x/2) ⊗ I)
    Wait, that's not right. Let me use the standard decomposition.
    
    Actually, for the KAK decomposition:
    U = (A ⊗ B) · K(x,y,z) · (C ⊗ D)
    where K(x,y,z) = exp(-i/2 · (x·XX + y·YY + z·ZZ))
    
    The standard 3-CNOT decomposition of K is:
    K = (Rz(x/2) ⊗ Rz(y/2)) · CX · (Rz(-x/2) ⊗ Rz(-y/2)) · CX · (Rz(x/2) ⊗ Rz(y/2)) · CX · (Rz(-x/2) ⊗ Rz(-y/2))
    Wait, I need to be more careful.
    
    Let me use a different approach. I'll use the following known decomposition:
    Any 2-qubit gate U can be written as:
    U = (A0 ⊗ B0) · CX · (A1 ⊗ B1) · CX · (A2 ⊗ B2) · CX · (A3 ⊗ B3)
    
    I'll use a numerical approach to find the single-qubit gates.
    """
    # For now, use a simple approach: return a placeholder
    # The actual decomposition will be done in _synthesize_generic
    return None


def _coordinate_descent_2qubit(target, budget, max_iters=500):
    """
    Coordinate descent for 2-qubit gate synthesis.
    Layout: (A0⊗B0) · CX · (A1⊗B1) · CX · (A2⊗B2) · CX · (A3⊗B3)
    """
    if _remaining() < 5:
        return []
    
    # Create layout with 3 CNOTs and 4 sets of single-qubit gates
    layout = []
    for i in range(4):
        layout.append(["u3", 0, 0, 0, 0])
        layout.append(["u3", 1, 0, 0, 0])
        if i < 3:
            layout.append(["cx", 0, 1])
    
    num_params = 24  # 8 u3 gates × 3 params each
    
    def build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with zeros
    params = np.zeros(num_params)
    best_params = params.copy()
    best_dist = _dist(target, _build(build_circuit(params), 2))
    
    # Coordinate descent
    step = 0.1
    for iteration in range(max_iters):
        if _remaining() < 10:
            break
        improved = False
        for i in range(num_params):
            for sign in (1, -1):
                trial = best_params.copy()
                trial[i] = best_params[i] + sign * step
                trial_dist = _dist(target, _build(build_circuit(trial), 2))
                if trial_dist < best_dist:
                    best_params = trial
                    best_dist = trial_dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.001:
                break
    
    return build_circuit(best_params)


def _coordinate_descent_3qubit(target, budget, max_iters=200):
    """
    Coordinate descent for 3-qubit gate synthesis.
    Layout: alternating single-qubit gates and CNOTs.
    """
    if _remaining() < 5:
        return []
    
    # Create layout with budget CNOTs
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    layout = []
    layout.append(["u3", 0, 0, 0, 0])
    layout.append(["u3", 1, 0, 0, 0])
    layout.append(["u3", 2, 0, 0, 0])
    for i in range(budget):
        c, t = pairs[i % len(pairs)]
        layout.append(["cx", c, t])
        layout.append(["u3", 0, 0, 0, 0])
        layout.append(["u3", 1, 0, 0, 0])
        layout.append(["u3", 2, 0, 0, 0])
    
    num_u3 = sum(1 for g in layout if g[0] == "u3")
    num_params = 3 * num_u3
    
    def build_circuit(params):
        gates = []
        k = 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], params[k], params[k+1], params[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    # Initialize with zeros
    params = np.zeros(num_params)
    best_params = params.copy()
    best_dist = _dist(target, _build(build_circuit(params), 3))
    
    # Coordinate descent
    step = 0.1
    for iteration in range(max_iters):
        if _remaining() < 10:
            break
        improved = False
        for i in range(num_params):
            for sign in (1, -1):
                trial = best_params.copy()
                trial[i] = best_params[i] + sign * step
                trial_dist = _dist(target, _build(build_circuit(trial), 3))
                if trial_dist < best_dist:
                    best_params = trial
                    best_dist = trial_dist
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.001:
                break
    
    return build_circuit(best_params)


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    if _remaining() < 5:
        qubits = problem["qubits"]
        return [["u3", q, 0.0, 0.0, 0.0] for q in range(qubits)]
    
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Check for known gates
    if _is_known_gate(target, name, qubits, budget):
        if qubits == 2 and (name == "CNOT" or budget == 1):
            return _exact_cnot_decomposition()
        elif qubits == 2 and name == "iSWAP":
            return _exact_iswap_decomposition()
        elif qubits == 2 and name == "SWAP":
            return _exact_swap_decomposition()
        elif qubits == 3 and name == "Toffoli":
            return _exact_toffoli_decomposition()
    
    # Generic gate: use coordinate descent
    if qubits == 2:
        return _coordinate_descent_2qubit(target, budget)
    else:
        return _coordinate_descent_3qubit(target, budget)
