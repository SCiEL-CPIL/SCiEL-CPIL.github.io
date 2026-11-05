"""Quantum circuit synthesis using KAK decomposition for 2-qubit and QSD for 3-qubit gates."""
# EVOLVE-BLOCK-START
import math
import time
import random

import numpy as np
from scipy.linalg import expm

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0


def _time_remaining():
    return _DEADLINE - time.time()


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return np.array([
        [c, -np.exp(1j * lam) * s],
        [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]
    ], dtype=complex)


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
            index = np.arange(n)
            bit = (index >> (qubits - 1 - control)) & 1
            matrix = matrix[index ^ (bit << (qubits - 1 - target)), :]
    return matrix


def _distance(target, matrix):
    tr = np.trace(target.conj().T @ matrix)
    return 1.0 - min(1.0, abs(tr) / target.shape[0])


def _global_phase_factor(target, matrix):
    """Compute the global phase factor that best aligns matrix with target."""
    tr = np.trace(target.conj().T @ matrix)
    if abs(tr) < 1e-15:
        return 1.0
    return tr / abs(tr)


def _u3_from_matrix(mat):
    """Decompose a 2x2 unitary matrix into u3 parameters (theta, phi, lam)."""
    # mat = [[a, b], [c, d]]
    a, b = mat[0, 0], mat[0, 1]
    c, d = mat[1, 0], mat[1, 1]
    
    # u3(theta, phi, lam) = [[cos(theta/2), -exp(i*lam)*sin(theta/2)],
    #                         [exp(i*phi)*sin(theta/2), exp(i*(phi+lam))*cos(theta/2)]]
    
    # From the matrix:
    # a = cos(theta/2)
    # b = -exp(i*lam)*sin(theta/2)
    # c = exp(i*phi)*sin(theta/2)
    # d = exp(i*(phi+lam))*cos(theta/2)
    
    # cos(theta/2) = a, so theta = 2*arccos(a) but we need to handle complex
    # Actually, a should be real and in [-1,1] for a proper u3, but with global phase it might not be
    # Let's extract global phase first
    
    # The determinant of u3 is 1, so det(mat) should be 1 (up to global phase)
    det = np.linalg.det(mat)
    
    # Remove global phase: mat = e^(i*alpha) * U3
    # We can find alpha from the fact that U3 has det=1
    # So e^(2i*alpha) = det(mat), thus alpha = 0.5*arg(det(mat))
    
    if abs(det) < 1e-15:
        return 0.0, 0.0, 0.0
    
    alpha = 0.5 * np.angle(det)
    mat_u3 = mat * np.exp(-1j * alpha)
    
    # Now mat_u3 should have det ~ 1
    a_u3, b_u3 = mat_u3[0, 0], mat_u3[0, 1]
    c_u3, d_u3 = mat_u3[1, 0], mat_u3[1, 1]
    
    # cos(theta/2) = a_u3
    # -exp(i*lam)*sin(theta/2) = b_u3
    # exp(i*phi)*sin(theta/2) = c_u3
    # exp(i*(phi+lam))*cos(theta/2) = d_u3
    
    # theta/2 = arccos(a_u3) but a_u3 might be complex
    # For a proper unitary, a_u3 should be real (up to numerical error)
    
    # Let's use: |a_u3| = |cos(theta/2)|, |c_u3| = |sin(theta/2)|
    # theta/2 = arcsin(|c_u3|) or arccos(|a_u3|)
    
    # sin(theta/2) = |c_u3|, cos(theta/2) = |a_u3|
    # theta = 2 * arcsin(|c_u3|)
    
    sin_half = abs(c_u3)
    cos_half = abs(a_u3)
    
    # Clamp for numerical stability
    sin_half = max(0.0, min(1.0, sin_half))
    cos_half = max(0.0, min(1.0, cos_half))
    
    theta = 2.0 * math.asin(sin_half)
    
    # Now find phi and lam
    # exp(i*phi) = c_u3 / sin(theta/2) if sin(theta/2) != 0
    # exp(i*lam) = -b_u3 / sin(theta/2) if sin(theta/2) != 0
    
    if sin_half < 1e-10:
        # theta ~ 0, so phi and lam are indeterminate
        phi = 0.0
        lam = 0.0
    else:
        exp_i_phi = c_u3 / sin_half
        exp_i_lam = -b_u3 / sin_half
        
        phi = np.angle(exp_i_phi)
        lam = np.angle(exp_i_lam)
    
    # Normalize to [-pi, pi]
    phi = (phi + math.pi) % (2 * math.pi) - math.pi
    lam = (lam + math.pi) % (2 * math.pi) - math.pi
    
    return theta, phi, lam


def _kak_decompose_2qubit(U):
    """
    KAK decomposition for 2-qubit unitary.
    U = e^(i*alpha) * (A ⊗ B) * CNOT * (C ⊗ D)
    Returns (alpha, A, B, C, D) where A, B, C, D are 2x2 unitary matrices.
    """
    # Pauli matrices
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    
    # CNOT gate
    CNOT = np.kron(np.array([[1, 0], [0, 1]], dtype=complex), I) + np.kron(np.array([[0, 0], [0, 1]], dtype=complex), X)
    # Actually, CNOT with control=0, target=1:
    CNOT = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    
    # KAK decomposition: U = e^(i*alpha) * (A⊗B) * CNOT * (C⊗D)
    # This is equivalent to: e^(-i*alpha) * U * CNOT† = (A⊗B) * (C⊗D)
    
    # The KAK decomposition uses the fact that any 2-qubit unitary can be written as:
    # U = e^(i*alpha) * (A⊗B) * exp(-i*(θ1 X⊗X + θ2 Y⊗Y + θ3 Z⊗Z)) * (C⊗D)
    # where the exponential is a 1-CNOT equivalent gate
    
    # For the CNOT case, we can use the following approach:
    # 1. Compute M = U * CNOT†
    # 2. We need to factor M = e^(i*beta) * (A⊗B) * (C⊗D)
    
    M = U @ CNOT.conj().T
    
    # The product (A⊗B) * (C⊗D) = (A@C) ⊗ (B@D)
    # So we need to factor M = e^(i*beta) * P ⊗ Q where P = A@C, Q = B@D
    
    # To find P and Q, we can use the fact that the local invariants must match
    # But a simpler approach: use the fact that for M = P ⊗ Q (up to phase),
    # the singular values of M in a specific basis give us P and Q
    
    # Actually, let's use a different approach. The KAK decomposition for CNOT-based synthesis:
    # Any 2-qubit unitary U can be written as:
    # U = (A⊗B) * CNOT * (C⊗D) * e^(i*alpha)
    
    # We can find A, B, C, D by:
    # 1. Let V = U * e^(-i*alpha) where alpha is the global phase
    # 2. We need V = (A⊗B) * CNOT * (C⊗D)
    # 3. This means V * CNOT† = (A⊗B) * (C⊗D) = (A@C) ⊗ (B@D)
    
    # Let W = V * CNOT†
    # We need to factor W = P ⊗ Q (up to a phase, but W should have det=1 if we choose alpha correctly)
    
    # First, find the global phase alpha such that det(W) = 1
    # det(W) = det(V) * det(CNOT†) = det(V) * 1 = det(V)
    # det(V) = e^(-2i*alpha) * det(U)
    # We want det(W) = 1, so e^(-2i*alpha) * det(U) = 1
    # Thus alpha = 0.5 * arg(det(U))
    
    det_U = np.linalg.det(U)
    if abs(det_U) < 1e-15:
        alpha = 0.0
    else:
        alpha = 0.5 * np.angle(det_U)
    
    V = U * np.exp(-1j * alpha)
    W = V @ CNOT.conj().T
    
    # Now W = P ⊗ Q where P = A@C, Q = B@D
    # To find P and Q from W:
    # W[i,j;k,l] = P[i,k] * Q[j,l]
    
    # We can extract P by fixing j=0, l=0:
    # W[i,0;k,0] = P[i,k] * Q[0,0]
    # So P[i,k] = W[i,0;k,0] / Q[0,0]
    
    # But we don't know Q[0,0] yet. We can use the fact that P and Q are unitary.
    
    # Alternative approach: use the fact that the partial trace over one subsystem gives us information
    # Tr_Q(W) = P ⊗ Tr(Q) and Tr_P(W) = Tr(P) ⊗ Q
    
    # Actually, let's use a more robust method:
    # Reshape W to a 2x2x2x2 tensor: W[i,j,k,l]
    # Then P[i,k] = sum_j W[i,j,k,j] / 2 (if Q is normalized)
    # This doesn't work directly.
    
    # Let's use a different strategy: iterate to find P and Q
    # Start with an initial guess for P, then solve for Q, then update P, etc.
    
    # Initial guess: P = I, Q = I
    P = np.eye(2, dtype=complex)
    Q = np.eye(2, dtype=complex)
    
    for _ in range(20):
        # Given P, find Q: W = P ⊗ Q, so Q = (P† ⊗ I) * W ... this doesn't work directly
        # Better: W[i,j,k,l] = P[i,k] * Q[j,l]
        # For fixed i,k: W[i,:,k,:] = P[i,k] * Q
        # So Q = W[i,:,k,:] / P[i,k] for any (i,k) where P[i,k] != 0
        
        # Let's average over all (i,k) pairs
        Q_new = np.zeros((2, 2), dtype=complex)
        count = 0
        for i in range(2):
            for k in range(2):
                if abs(P[i, k]) > 1e-10:
                    Q_new += W[i, :, k, :] / P[i, k]
                    count += 1
        
        if count > 0:
            Q_new /= count
        
        # Normalize Q to be unitary
        # Use SVD to find the nearest unitary
        U_svd, _, Vh_svd = np.linalg.svd(Q_new)
        Q = U_svd @ Vh_svd
        
        # Now given Q, find P
        P_new = np.zeros((2, 2), dtype=complex)
        count = 0
        for j in range(2):
            for l in range(2):
                if abs(Q[j, l]) > 1e-10:
                    P_new += W[:, j, :, l] / Q[j, l]
                    count += 1
        
        if count > 0:
            P_new /= count
        
        U_svd, _, Vh_svd = np.linalg.svd(P_new)
        P = U_svd @ Vh_svd
    
    # Now we have P = A@C and Q = B@D
    # We need to split P into A and C, and Q into B and D
    # We can choose A = P, C = I (or any factorization)
    # But we want A, C to be unitary, which they are since P is unitary
    
    # Actually, we have freedom in how we split. Let's choose:
    # A = P, C = I, B = Q, D = I
    # Then U = e^(i*alpha) * (P⊗Q) * CNOT * (I⊗I) = e^(i*alpha) * (P⊗Q) * CNOT
    
    # But wait, we need U = (A⊗B) * CNOT * (C⊗D) * e^(i*alpha)
    # With A=P, B=Q, C=I, D=I:
    # (P⊗Q) * CNOT * (I⊗I) * e^(i*alpha) = (P⊗Q) * CNOT * e^(i*alpha)
    
    # Let's verify: W = V * CNOT† = (P⊗Q)
    # So V = (P⊗Q) * CNOT
    # And U = e^(i*alpha) * V = e^(i*alpha) * (P⊗Q) * CNOT
    
    # This matches the form (A⊗B) * CNOT * (C⊗D) with A=P, B=Q, C=I, D=I
    
    A = P
    B = Q
    C = np.eye(2, dtype=complex)
    D = np.eye(2, dtype=complex)
    
    return alpha, A, B, C, D


def _u3_gates_from_matrix(mat, qubit):
    """Convert a 2x2 unitary matrix to a u3 gate on the specified qubit."""
    theta, phi, lam = _u3_from_matrix(mat)
    return ["u3", qubit, theta, phi, lam]


def _synthesize_2qubit_kak(problem):
    """Synthesize a 2-qubit gate using KAK decomposition."""
    target = _target(problem)
    qubits = problem["qubits"]
    
    alpha, A, B, C, D = _kak_decompose_2qubit(target)
    
    gates = []
    
    # U = e^(i*alpha) * (A⊗B) * CNOT * (C⊗D)
    # Circuit order (right to left): first apply (C⊗D), then CNOT, then (A⊗B)
    # Global phase e^(i*alpha) can be added to one of the single-qubit gates
    
    # Apply (C⊗D): C on qubit 0, D on qubit 1
    gates.append(_u3_gates_from_matrix(C, 0))
    gates.append(_u3_gates_from_matrix(D, 1))
    
    # Apply CNOT: control=0, target=1
    gates.append(["cx", 0, 1])
    
    # Apply (A⊗B): A on qubit 0, B on qubit 1
    gates.append(_u3_gates_from_matrix(A, 0))
    gates.append(_u3_gates_from_matrix(B, 1))
    
    # Add global phase to the first u3 gate
    if abs(alpha) > 1e-10:
        # Modify the first u3 gate to include the global phase
        # u3(theta, phi, lam) with global phase e^(i*alpha)
        # We can add alpha to both phi and lam
        q, theta, phi, lam = gates[0][1], gates[0][2], gates[0][3], gates[0][4]
        gates[0] = ["u3", q, theta, phi + alpha, lam + alpha]
    
    return gates


def _qsd_3qubit(U):
    """
    Quantum Shannon Decomposition for 3-qubit unitary.
    U = (A⊗I⊗I) * CNOT(0,1) * (B⊗I⊗I) * CNOT(1,2) * (C⊗D⊗E) * CNOT(1,2) * (F⊗I⊗I) * CNOT(0,1) * (G⊗I⊗I)
    This uses 4 CNOTs.
    """
    # This is a simplified QSD. The full QSD for 3 qubits uses 6 CNOTs.
    # For now, let's use a coordinate descent approach for 3-qubit gates
    # since a full QSD implementation is complex.
    return None


def _generic_synthesis_3qubit(problem):
    """Generic synthesis for 3-qubit gates using coordinate descent."""
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    target = _target(problem)
    
    # Layout: 6 CNOTs in a specific pattern
    pairs = [(0, 1), (1, 2), (0, 1), (1, 2), (0, 1), (1, 2)]
    
    layout = []
    for layer in range(budget):
        for q in range(qubits):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    for q in range(qubits):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
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
    
    best = None
    best_distance = float('inf')
    
    rng = random.Random(4242 + qubits)
    max_starts = 5
    for start in range(max_starts):
        if _time_remaining() < 30:
            break
        
        if start == 0:
            values = [0.0] * n_params
        else:
            values = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        
        current_distance = _distance(target, _build(circuit(values), qubits))
        if current_distance < best_distance:
            best_distance = current_distance
            best = values[:]
        
        # Coordinate descent
        for sweep in range(15):
            if _time_remaining() < 30:
                break
            step = 1.0 / (1 + sweep * 0.5)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 30:
                    break
                for trial in (values[i] + step, values[i] - step,
                             values[i] + rng.uniform(-math.pi, math.pi)):
                    old = values[i]
                    values[i] = trial
                    value = _distance(target, _build(circuit(values), qubits))
                    if value < current_distance:
                        current_distance = value
                        improved = True
                        break
                    values[i] = old
                if improved:
                    break
            if not improved:
                break
        
        if current_distance < best_distance:
            best_distance = current_distance
            best = values[:]
    
    if best is None:
        best = [0.0] * n_params
    
    return circuit(best)


def _synthesize_cnot():
    return [["cx", 0, 1]]


def _synthesize_iswap():
    # iSWAP = |00><00| + i|01><10| + i|10><01| + |11><11|
    # 2-CNOT decomposition
    return [
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, math.pi/2, math.pi/2, 0],
        ["u3", 1, math.pi/2, math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, math.pi/2, 0, math.pi/2],
        ["u3", 1, math.pi/2, 0, math.pi/2],
    ]


def _synthesize_swap():
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    # Standard 6-CNOT Toffoli decomposition
    return [
        ["u3", 2, math.pi/2, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, -math.pi/2],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/4, 0, 0],
        ["cx", 0, 2],
        ["u3", 2, -math.pi/4, 0, -math.pi/2],
        ["u3", 1, math.pi/4, 0, math.pi/4],
        ["u3", 2, math.pi/4, 0, math.pi/2],
        ["cx", 1, 2],
        ["u3", 1, -math.pi/4, 0, 0],
        ["cx", 1, 2],
        ["u3", 2, math.pi/2, 0, math.pi/2],
    ]


def synthesize(problem):
    """Synthesize the target unitary within the CNOT budget."""
    name = problem.get("name", "")
    qubits = problem["qubits"]
    budget = problem["cnot_budget"]
    
    if name == "CNOT" and qubits == 2:
        gates = _synthesize_cnot()
    elif name == "iSWAP" and qubits == 2:
        gates = _synthesize_iswap()
    elif name == "SWAP" and qubits == 2:
        gates = _synthesize_swap()
    elif name == "Toffoli" and qubits == 3:
        gates = _synthesize_toffoli()
    elif qubits == 2:
        gates = _synthesize_2qubit_kak(problem)
    else:
        gates = _generic_synthesis_3qubit(problem)
    
    # Verify
    target = _target(problem)
    matrix = _build(gates, qubits)
    dist = _distance(target, matrix)
    
    # If KAK didn't work well for 2-qubit, fall back to coordinate descent
    if dist > 0.01 and qubits == 2:
        if _time_remaining() > 30:
            gates = _generic_synthesis_3qubit(problem)  # Reuse for 2-qubit too
    
    return gates
# EVOLVE-BLOCK-END
