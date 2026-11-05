"""Quantum circuit synthesis using analytical decompositions with CNOT budget constraints."""
import math
import time

import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds before the 270s limit to stop work


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_unitary(gates, qubits):
    dim = 2 ** qubits
    U = np.eye(dim, dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    for gate in gates:
        if gate[0] == "u3":
            _, q, theta, phi, lam = gate
            single = _u3_matrix(theta, phi, lam)
            # Build full operator via kron
            op = np.array([[1]], dtype=complex)
            for k in range(qubits):
                if k == q:
                    op = np.kron(op, single)
                else:
                    op = np.kron(op, eye2)
            U = op @ U
        else:
            _, control, target = gate
            # CNOT: flip target if control is 1
            # qubit 0 is most significant
            # Build CNOT operator
            cnot_op = np.eye(dim, dtype=complex)
            for i in range(dim):
                ctrl_bit = (i >> (qubits - 1 - control)) & 1
                if ctrl_bit == 1:
                    tgt_bit_pos = qubits - 1 - target
                    j = i ^ (1 << tgt_bit_pos)
                    cnot_op[j, i] = 1.0
            U = cnot_op @ U
    return U


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _decompose_2qubit(U):
    """Decompose a 2-qubit unitary into CNOTs and single-qubit gates.
    Returns (left, a, b, c, right, n_cnots) where:
    U = left @ (I⊗a) @ CNOT @ (I⊗b) @ CNOT @ (I⊗c) @ CNOT @ right
    or with fewer CNOTs.
    """
    # For 2 qubits, use the standard KAK decomposition
    # U = (A ⊗ B) CNOT^k (C ⊗ D)
    # We'll use a simpler approach: try to find the decomposition
    
    # Actually, let's use a direct approach for known gates
    # and a general fallback
    
    # For a general 2-qubit unitary, we can use the following method:
    # 1. Remove global phase
    # 2. Use KAK decomposition
    
    # Let's implement a simpler version that works for the specific gates we need
    
    # First, check if it's close to identity
    if np.allclose(U, np.eye(4), atol=1e-10):
        return [np.eye(2, dtype=complex), np.eye(2, dtype=complex),
                np.eye(2, dtype=complex), np.eye(2, dtype=complex),
                np.eye(2, dtype=complex), np.eye(2, dtype=complex), 0]
    
    # Check if it's a CNOT (or similar)
    # CNOT with control 0, target 1:
    cnot_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    if np.allclose(U, cnot_01, atol=1e-10):
        # U = CNOT = (I⊗I) CNOT (I⊗I) CNOT (I⊗I) CNOT
        # But that's 3 CNOTs. Actually CNOT is just 1 CNOT.
        # In the form U = L CNOT^k R, we can have k=1
        # U = (I) CNOT (I) with appropriate left/right
        # Actually: CNOT = (I⊗I) CNOT (I⊗I)
        # So left = I, right = I, and we need 1 CNOT with a=I, b=I, c=I
        # But the form is left @ (I⊗a) @ CNOT @ (I⊗b) @ CNOT @ (I⊗c) @ CNOT @ right
        # For 1 CNOT: left @ (I⊗a) @ CNOT @ right
        # We can set: left = I, a = I, right = I, and use 1 CNOT
        return [np.eye(2, dtype=complex), np.eye(2, dtype=complex),
                np.eye(2, dtype=complex), np.eye(2, dtype=complex),
                np.eye(2, dtype=complex), np.eye(2, dtype=complex), 1]
    
    # For now, return a placeholder that uses 3 CNOTs
    # We'll refine this later
    return [np.eye(2, dtype=complex), np.eye(2, dtype=complex),
            np.eye(2, dtype=complex), np.eye(2, dtype=complex),
            np.eye(2, dtype=complex), np.eye(2, dtype=complex), 3]


def _u3_from_matrix(M):
    """Convert a 2x2 unitary matrix to u3(theta, phi, lam) parameters."""
    # M = [[a, b], [c, d]]
    # u3: [[cos(t/2), -e^{iλ}sin(t/2)], [e^{iφ}sin(t/2), e^{i(φ+λ)}cos(t/2)]]
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    
    # Handle special cases
    if abs(a) < 1e-15 and abs(c) < 1e-15:
        # Zero column, shouldn't happen for unitary
        return 0.0, 0.0, 0.0
    
    # From the u3 form:
    # cos(t/2) = |a|
    # sin(t/2) = |b| = |c|
    # If sin(t/2) ≈ 0, then t ≈ 0, and we need to handle phases
    if abs(math.sin(math.acos(min(1.0, max(-1.0, abs(a)))))) < 1e-15:
        # t ≈ 0 or π
        if abs(a) > 0.999:
            # t ≈ 0
            # M ≈ [[e^{iγ}, 0], [0, e^{i(γ+δ)}]] approximately
            # From u3 with t=0: [[1, 0], [0, e^{i(φ+λ)}]]
            # So φ+λ = arg(d), and we can set φ=0, λ=arg(d)
            lam = np.angle(d)
            return 0.0, 0.0, lam
        else:
            # t ≈ π
            # M ≈ [[0, -e^{iλ}], [e^{iφ}, 0]]
            # λ = -arg(b), φ = arg(c)
            lam = -np.angle(b) if abs(b) > 1e-15 else 0.0
            phi = np.angle(c) if abs(c) > 1e-15 else 0.0
            return math.pi, phi, lam
    
    t = 2 * math.acos(min(1.0, max(-1.0, abs(a))))
    s = math.sin(t / 2)
    
    # -e^{iλ} sin(t/2) = b => e^{iλ} = -b/s
    lam = np.angle(-b / s)
    # e^{iφ} sin(t/2) = c => e^{iφ} = c/s
    phi = np.angle(c / s)
    
    return t, phi, lam


def _mat_to_u3(M):
    """Convert 2x2 unitary to u3 params, handling global phase."""
    # Remove global phase for cleaner decomposition
    # But we need to track it... For now, just decompose directly
    return _u3_from_matrix(M)


def _decompose_3qubit_to_cnots(U, qubits=3):
    """
    Decompose a 3-qubit unitary into CNOTs and single-qubit gates.
    Returns a list of gates.
    """
    dim = 2 ** qubits
    gates = []
    
    # Strategy: iteratively decompose
    # For 3 qubits, we can use the following approach:
    # 1. Use single-qubit gates to reduce the problem
    # 2. Use CNOTs to entangle
    
    # A simpler approach: use the fact that any 3-qubit unitary can be
    # decomposed with at most 15 CNOTs (known result)
    # But we need to respect the budget.
    
    # For now, return a placeholder
    # We'll implement a proper decomposition later
    return gates


def _known_gate_decompositions(name, qubits, target):
    """Return exact decompositions for known gates."""
    gates = []
    
    if name == "CNOT" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex), atol=1e-10)):
        # CNOT: control 0, target 1
        gates = [["cx", 0, 1]]
        return gates
    
    if name == "SWAP" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex), atol=1e-10)):
        # SWAP = CNOT(0,1) CNOT(1,0) CNOT(0,1)
        gates = [["cx", 0, 1], ["cx", 1, 0], ["cx", 0, 1]]
        return gates
    
    if name == "iSWAP" or (qubits == 2 and np.allclose(target, np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=complex), atol=1e-10)):
        # iSWAP decomposition
        # iSWAP = (H⊗H) CNOT (Rz(π/4)⊗Rz(-π/4)) CNOT (Rz(-π/4)⊗Rz(π/4)) CNOT (H⊗H)
        # More precisely, a standard decomposition uses 3 CNOTs
        # iSWAP = (S⊗S) CNOT (S†⊗S†) CNOT (S⊗S) CNOT with appropriate rotations
        # Let me use a known decomposition:
        # iSWAP = [Ry(π/2) Rx(π/2)] ⊗ [Ry(π/2) Rx(π/2)] CNOT ...
        
        # A cleaner decomposition:
        # iSWAP = (H ⊗ H) CNOT (Rz(π/4) ⊗ Rz(-π/4)) CNOT (Rz(-π/4) ⊗ Rz(π/4)) CNOT (H ⊗ H)
        # Let's use u3 gates:
        # H = u3(π/2, 0, π)
        # Rz(θ) = u3(0, 0, θ)
        
        H = (math.pi / 2, 0.0, math.pi)
        gates = [
            ["u3", 0, H[0], H[1], H[2]],
            ["u3", 1, H[0], H[1], H[2]],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, math.pi / 4],
            ["u3", 1, 0, 0, -math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, 0, 0, -math.pi / 4],
            ["u3", 1, 0, 0, math.pi / 4],
            ["cx", 0, 1],
            ["u3", 0, H[0], H[1], H[2]],
            ["u3", 1, H[0], H[1], H[2]],
        ]
        return gates
    
    if name == "Toffoli" or (qubits == 3 and np.allclose(target, np.diag([1,1,1,1,1,1,1,-1]).astype(complex), atol=1e-10)):
        # Toffoli decomposition with 6 CNOTs
        # Standard decomposition:
        gates = [
            ["u3", 2, 0, 0, math.pi / 8],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 8],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, math.pi / 8],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 8],
            ["u3", 2, 0, 0, math.pi],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, math.pi / 8],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 8],
            ["cx", 0, 2],
            ["u3", 2, 0, 0, math.pi / 8],
            ["cx", 1, 2],
            ["u3", 2, 0, 0, -math.pi / 8],
        ]
        return gates
    
    return None


def synthesize(problem):
    """Synthesize a quantum circuit for the given target unitary."""
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 20s margin
    
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    name = problem.get("name", "")
    
    # Check for known gates
    known = _known_gate_decompositions(name, qubits, target)
    if known is not None:
        # Verify it's within budget
        cnot_count = sum(1 for g in known if g[0] == "cx")
        if cnot_count <= budget:
            return known
    
    # For generic unitaries, use iterative decomposition
    # Start with identity and build up
    
    # For 2-qubit generic unitaries, use KAK-based decomposition
    if qubits == 2:
        return _synthesize_2qubit(target, budget, deadline)
    
    # For 3-qubit generic unitaries
    return _synthesize_3qubit(target, budget, deadline)


def _synthesize_2qubit(target, budget, deadline):
    """Synthesize a 2-qubit unitary within CNOT budget."""
    # Use the fact that any 2-qubit unitary can be decomposed with ≤3 CNOTs
    # U = (A ⊗ B) CNOT^k (C ⊗ D)
    
    # For k=3: U = (A⊗B) CNOT (C⊗D) CNOT (E⊗F) CNOT (G⊗H)
    # We need to find A, B, C, D, E, F, G, H
    
    # Simplified approach: use the standard decomposition
    # U = (L0 ⊗ L1) CNOT (M0 ⊗ M1) CNOT (N0 ⊗ N1) CNOT (R0 ⊗ R1)
    
    # Let's use a numerical approach:
    # 1. Assume 3 CNOTs
    # 2. Optimize the single-qubit rotations
    
    # Initialize with identity
    params = [0.0] * 24  # 8 matrices × 3 params each
    
    # For now, return a simple decomposition
    # We'll improve this with optimization if time permits
    
    gates = []
    # Placeholder: return all-identity with 3 CNOTs
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 0, 1])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 0, 1])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    gates.append(["cx", 0, 1])
    for q in range(2):
        gates.append(["u3", q, 0.0, 0.0, 0.0])
    
    return gates


def _synthesize_3qubit(target, budget, deadline):
    """Synthesize a 3-qubit unitary within CNOT budget."""
    # For 3-qubit, use up to 15 CNOTs (known upper bound)
    # But respect the budget
    
    gates = []
    # Placeholder
    return gates
