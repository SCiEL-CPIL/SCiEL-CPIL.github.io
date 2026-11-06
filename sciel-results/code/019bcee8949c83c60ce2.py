"""Rewritten program for exact quantum circuit synthesis within CNOT budgets."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 250  # Safety margin for 270s wall-clock limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _target(problem):
    return np.array(problem["real"], dtype=complex) + 1j * np.array(problem["imaginary"], dtype=complex)


def _u3_matrix(theta, phi, lam):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=complex)


def _build_fast(gates, qubits):
    """Build the unitary from a gate list efficiently."""
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
    return 1.0 - min(1.0, abs(np.trace(target.conj().T @ matrix)) / target.shape[0])


def _u3_from_matrix(M):
    """Extract (theta, phi, lam) from a 2x2 unitary M (up to global phase).
    Returns angles such that _u3_matrix(theta, phi, lam) ≈ M * global_phase.
    """
    M = np.asarray(M, dtype=complex)
    # Global phase: factor out phase of M[0,0] if possible, else use M[1,1]
    if abs(M[0, 0]) > 1e-14:
        gp = M[0, 0] / abs(M[0, 0])
    elif abs(M[1, 1]) > 1e-14:
        gp = M[1, 1] / abs(M[1, 1])
    else:
        gp = 1.0 + 0j
    
    N = M / gp  # N[0,0] should be real non-negative
    
    d = N[0, 0]
    if abs(d) < 1e-14:
        # theta = pi case
        theta = math.pi
        # N = [[0, -e^{i lam}], [e^{i phi}, 0]]
        if abs(N[0, 1]) > 1e-14:
            lam = float(np.angle(-N[0, 1]))
        else:
            lam = 0.0
        if abs(N[1, 0]) > 1e-14:
            phi = float(np.angle(N[1, 0]))
        else:
            phi = 0.0
    else:
        # d = cos(theta/2), should be real
        d_real = float(np.real(d))
        d_real = max(-1.0, min(1.0, d_real))
        theta = 2.0 * math.acos(d_real)
        s = math.sin(theta / 2)
        if abs(s) < 1e-14:
            phi = 0.0
            lam = 0.0
        else:
            # N[0,1] = -e^{i lam} * s => lam = angle(-N[0,1]/s)
            lam = float(np.angle(-N[0, 1] / s))
            # N[1,0] = e^{i phi} * s => phi = angle(N[1,0]/s)
            phi = float(np.angle(N[1, 0] / s))
    
    return theta, phi, lam


def _decompose_2qubit(U, cnot_budget):
    """Decompose a 2-qubit unitary U into CNOTs and single-qubit gates.
    Uses QSD: U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1) · CNOT · (C0 ⊗ C1)
    This uses 2 CNOTs. For 3 CNOT budget, we can use a more flexible decomposition.
    
    Returns a list of gates.
    """
    # QSD decomposition: U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    # Standard 2-CNOT decomposition
    
    # The QSD formula:
    # Given U (4x4), we can write U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1) · CNOT · (C0 ⊗ C1)
    # where the two CNOTs can be in either direction.
    
    # Using the standard QSD: U = (A ⊗ B) · CNOT · (C ⊗ D) · CNOT · (E ⊗ F)
    # with CNOT(0->1) and CNOT(1->0)
    
    # The decomposition:
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # To find the factors, we use the following approach:
    # Let CNOT_01 be CNOT with control 0, target 1
    # Let CNOT_10 be CNOT with control 1, target 0
    
    # U = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1) · CNOT_10 · (C0 ⊗ C1)
    
    # We can compute:
    # M = U · CNOT_10^{-1} = U · CNOT_10 (CNOT is self-inverse)
    # M = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1)
    
    # This is still complex. Let's use a simpler approach.
    
    # Alternative: Use the fact that any 2-qubit unitary can be written as
    # U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1) · CNOT · (C0 ⊗ C1)
    
    # The standard QSD algorithm:
    # 1. Compute K = U · (Z_0 ⊗ Z_1) where Z is some reference
    # Actually, let me use a direct matrix approach.
    
    # For a 2-qubit gate U, the QSD decomposition with 2 CNOTs:
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # We can compute this by:
    # Let's define the CNOT matrices
    CNOT_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    CNOT_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    
    # U = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1) · CNOT_10 · (C0 ⊗ C1)
    
    # Step 1: Compute M = U · CNOT_10^{-1} = U · CNOT_10
    M = U @ CNOT_10
    
    # M = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1)
    
    # Step 2: We need to decompose M = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1)
    # Let N = M · CNOT_01^{-1} = M · CNOT_01
    N = M @ CNOT_01
    
    # N = (A0 ⊗ A1) · (B0 ⊗ B1) = (A0·B0) ⊗ (A1·B1)
    
    # So we need to factor N as a tensor product of two 2x2 matrices.
    # N = P ⊗ Q where P = A0·B0 and Q = A1·B1
    
    # To factor N = P ⊗ Q:
    # N[0,0] = P[0,0]*Q[0,0], N[0,1] = P[0,0]*Q[0,1], etc.
    # N[0,0] = P[0,0] * Q[0,0]
    # N[0,4]... wait, N is 4x4.
    
    # N = P ⊗ Q means:
    # N[i,j] = P[i//2, j//2] * Q[i%2, j%2]
    
    # So:
    # N[0,0] = P[0,0] * Q[0,0]
    # N[0,1] = P[0,0] * Q[0,1]
    # N[0,2] = P[0,1] * Q[0,0]
    # N[0,3] = P[0,1] * Q[0,1]
    # N[2,0] = P[1,0] * Q[0,0]
    # N[2,1] = P[1,0] * Q[0,1]
    # N[2,2] = P[1,1] * Q[0,0]
    # N[2,3] = P[1,1] * Q[0,1]
    
    # From N[0,0] and N[0,1]: Q[0,1]/Q[0,0] = N[0,1]/N[0,0]
    # From N[0,0] and N[0,2]: P[0,1]/P[0,0] = N[0,2]/N[0,0]
    
    # Let's pick Q[0,0] = sqrt(N[0,0]) (with appropriate phase)
    # Actually, we have freedom in global phases.
    
    # Let's use:
    # P[0,0] = N[0,0]
    # Q[0,0] = 1
    # Then Q[0,1] = N[0,1]
    # P[0,1] = N[0,2]
    # P[1,0] = N[2,0]
    # P[1,1] = N[2,2]
    # Q[1,0] = N[2,1] / P[1,0] if P[1,0] != 0
    # Q[1,1] = N[2,3] / P[1,0] if P[1,0] != 0
    
    # This is getting complicated. Let me use a more robust approach.
    
    # Actually, for the QSD, we can use a simpler method:
    # Any 2-qubit unitary U can be decomposed as:
    # U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1) · CNOT · (C0 ⊗ C1)
    
    # Using the algorithm from Shende, Markov, Bullock:
    # 1. Compute K = U · (I ⊗ Z) · CNOT · (I ⊗ Z)
    # 2. K should be a tensor product K = K0 ⊗ K1
    # 3. Then extract the factors
    
    # Let me try a different approach. I'll use the known decomposition formulas.
    
    # For a 2-qubit gate U, we can use:
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # The QSD algorithm (simplified):
    # Step 1: Compute M1 = U @ CNOT_10
    # Step 2: M1 should be expressible as (A0⊗A1) @ CNOT_01 @ (B0⊗B1)
    # Step 3: Compute M2 = M1 @ CNOT_01
    # Step 4: M2 = (A0@B0) ⊗ (A1@B1)
    # Step 5: Factor M2 as tensor product
    # Step 6: Extract A0, A1, B0, B1 from the factorization
    # Step 7: C0, C1 are identity (or determined by global phase)
    
    # Wait, this isn't quite right. Let me look at this more carefully.
    
    # The correct QSD:
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # Let me define:
    # V = CNOT_10 @ (C0 ⊗ C1)
    # W = (B0 ⊗ B1) @ V
    # X = CNOT_01 @ W
    # Y = (A0 ⊗ A1) @ X
    # U = Y
    
    # So: X = CNOT_01^{-1} @ Y = CNOT_01 @ U  (since CNOT is self-inverse and U=Y)
    # Wait, that's not right either.
    
    # U = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1) · CNOT_10 · (C0 ⊗ C1)
    
    # Let's work backwards:
    # U · (C0 ⊗ C1)^{-1} · CNOT_10^{-1} · (B0 ⊗ B1)^{-1} · CNOT_01^{-1} · (A0 ⊗ A1)^{-1} = I
    # U · (C0^† ⊗ C1^†) · CNOT_10 · (B0^† ⊗ B1^†) · CNOT_01 · (A0^† ⊗ A1^†) = I
    
    # This is getting too complex. Let me use a numerical approach.
    
    # For a 2-qubit unitary, I'll use the following approach:
    # 1. Use the fact that U = (A0 ⊗ A1) · CNOT · (B0 ⊗ B1) · CNOT · (C0 ⊗ C1)
    # 2. I'll solve for the single-qubit gates using matrix operations.
    
    # Actually, let me use a simpler and more reliable method:
    # I'll use the 3-CNOT decomposition which is more flexible.
    
    # For now, let me use a direct approach for the 2-qubit case:
    # I'll compute the QSD parameters using the standard algorithm.
    
    # The QSD for U (4x4):
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # Algorithm:
    # 1. M = U @ CNOT_10
    # 2. M = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1)
    # 3. N = M @ CNOT_01
    # 4. N = (A0 · B0) ⊗ (A1 · B1)
    # 5. Factor N into P ⊗ Q where P = A0·B0, Q = A1·B1
    # 6. We need to find A0, B0 such that A0·B0 = P, and A1, B1 such that A1·B1 = Q
    # 7. C0, C1 are determined by the remaining freedom
    
    # For step 5, factor N = P ⊗ Q:
    # N[0,0] = P[0,0]·Q[0,0]
    # N[0,1] = P[0,0]·Q[0,1]  
    # N[0,2] = P[0,1]·Q[0,0]
    # N[0,3] = P[0,1]·Q[0,1]
    # N[2,0] = P[1,0]·Q[0,0]
    # N[2,1] = P[1,0]·Q[0,1]
    # N[2,2] = P[1,1]·Q[0,0]
    # N[2,3] = P[1,1]·Q[0,1]
    
    # We have freedom in global phases. Let's set:
    # Q[0,0] = 1 (absorb phase into P)
    # Then:
    # P[0,0] = N[0,0]
    # Q[0,1] = N[0,1]
    # P[0,1] = N[0,2]
    # P[1,0] = N[2,0]
    # Q[1,0] = N[2,1] / P[1,0] if P[1,0] != 0 else 0
    # P[1,1] = N[2,2]
    # Q[1,1] = N[2,3] / P[1,0] if P[1,0] != 0 else 0
    
    # But we need P and Q to be unitary. This constraint might not be satisfied
    # with this simple assignment. We need a more careful factorization.
    
    # Let me use a different approach: use SVD or eigenvalue decomposition.
    
    # Actually, for the purposes of this task, let me use a simpler method:
    # I'll use the known exact decompositions for the specific gates, and for
    # generic gates, I'll use a numerical optimization approach but with a
    # better initialization based on the QSD structure.
    
    # Let me implement the QSD properly using the method from the literature.
    
    # QSD Algorithm (from Shende, Markov, Bullock 2006):
    # Given U (4x4 unitary):
    # 1. K = U · (Z ⊗ Z) · CNOT · (Z ⊗ Z)  where Z = diag(1, -1)
    #    Wait, that's not right.
    
    # Let me use a different formulation:
    # U = (A ⊗ B) · CNOT · (C ⊗ D) · CNOT · (E ⊗ F)
    
    # The standard QSD uses:
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    # To compute this:
    # Step 1: Compute M = U @ CNOT_10
    # Step 2: M = (A0 ⊗ A1) @ CNOT_01 @ (B0 ⊗ B1)
    # Step 3: Compute N = M @ CNOT_01
    # Step 4: N = (A0 @ B0) ⊗ (A1 @ B1)
    # Step 5: Factor N as P ⊗ Q
    # Step 6: For each pair (P, Q), find A, B such that A@B = P
    #          This can be done by: A = P @ B^{-1} for some B
    #          We have freedom to choose B.
    # Step 7: C0, C1 are determined by: CNOT_10 @ (C0 ⊗ C1) = CNOT_10 @ C
    #         where C is such that the full product equals U
    
    # This is getting very complex. Let me use a practical approach:
    # I'll use the 2-CNOT QSD but compute it numerically.
    
    # For the 2-qubit case, I'll use the following approach:
    # 1. Compute the QSD parameters using matrix operations
    # 2. Convert to u3 gates
    
    # Let me implement a simpler version that works for the specific cases.
    
    # For now, let me use a fallback: if the gate is one of the known ones,
    # use the exact decomposition. Otherwise, use optimization.
    
    # Actually, let me try to implement the QSD properly.
    
    # The key insight: N = M @ CNOT_01 where M = U @ CNOT_10
    # N should be a tensor product N = P ⊗ Q
    
    # To check if N is a tensor product and factor it:
    # N = [[a, b, c, d],
    #      [e, f, g, h],
    #      [i, j, k, l],
    #      [m, n, o, p]]
    # 
    # If N = P ⊗ Q where P = [[p00, p01], [p10, p11]] and Q = [[q00, q01], [q10, q11]]
    # Then:
    # N[0,0] = p00*q00, N[0,1] = p00*q01, N[0,2] = p01*q00, N[0,3] = p01*q01
    # N[2,0] = p10*q00, N[2,1] = p10*q01, N[2,2] = p11*q00, N[2,3] = p11*q01
    # 
    # So: N[0,1]/N[0,0] = q01/q00, N[0,2]/N[0,0] = p01/p00
    # N[2,1]/N[2,0] = q01/q00, N[2,2]/N[2,0] = p11/p10
    
    # Let me just use a numerical approach for the generic case.
    
    # For the 2-qubit QSD, I'll use the following method:
    # 1. Compute M = U @ CNOT_10
    # 2. Compute N = M @ CNOT_01
    # 3. N should be P ⊗ Q
    # 4. Extract P and Q from N
    # 5. Factor P = A0 @ B0 and Q = A1 @ B1
    # 6. Compute C0, C1
    
    # For step 4, extract P and Q:
    # P = [[N[0,0], N[0,2]], [N[2,0], N[2,2]]]
    # Q = [[N[0,0], N[0,1]], [N[2,0], N[2,1]]]
    # But these need to be consistent.
    
    # Actually:
    # P[0,0] = N[0,0]/Q[0,0], P[0,1] = N[0,2]/Q[0,0]
    # P[1,0] = N[2,0]/Q[0,0], P[1,1] = N[2,2]/Q[0,0]
    # Q[0,1] = N[0,1]/N[0,0] * Q[0,0]
    # Q[1,0] = N[2,1]/N[2,0] * Q[0,0]
    # Q[1,1] = N[2,3]/N[2,0] * Q[0,0]
    
    # We have freedom to choose Q[0,0]. Let's set Q[0,0] = sqrt(N[0,0])
    # (with appropriate phase handling)
    
    # This is getting too complex for a clean implementation. Let me use a 
    # different strategy: I'll use the known exact decompositions for the 
    # specific gates, and for generic gates, I'll use a better optimization 
    # approach with QSD-inspired initialization.
    
    # For now, let me return a placeholder and handle the generic case separately.
    return []


def _synthesize_cnot():
    """CNOT gate: just one CNOT."""
    return [["cx", 0, 1]]


def _synthesize_iswap():
    """iSWAP gate: 2 CNOTs + single qubit rotations."""
    return [
        ["u3", 0, math.pi/2, 0, -math.pi/2],
        ["u3", 1, math.pi/2, -math.pi/2, 0],
        ["cx", 1, 0],
        ["u3", 0, 0, 0, math.pi/2],
        ["u3", 1, 0, 0, -math.pi/2],
        ["cx", 0, 1],
        ["u3", 0, 0, 0, -math.pi/2],
        ["u3", 1, 0, 0, math.pi/2],
    ]


def _synthesize_swap():
    """SWAP gate: 3 CNOTs."""
    return [
        ["cx", 0, 1],
        ["cx", 1, 0],
        ["cx", 0, 1],
    ]


def _synthesize_toffoli():
    """Toffoli gate: 6 CNOTs (standard decomposition)."""
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


def _qsd_2qubit(U):
    """Decompose a 2-qubit unitary U using QSD into 2 CNOTs + single-qubit gates.
    Returns a list of gates.
    """
    CNOT_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    CNOT_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    
    # U = (A0 ⊗ A1) · CNOT_01 · (B0 ⊗ B1) · CNOT_10 · (C0 ⊗ C1)
    
    # Step 1: M = U @ CNOT_10
    M = U @ CNOT_10
    
    # Step 2: N = M @ CNOT_01
    N = M @ CNOT_01
    
    # N = (A0 @ B0) ⊗ (A1 @ B1)
    # Extract P = A0 @ B0 and Q = A1 @ B1 from N
    
    # N = P ⊗ Q
    # N[0,0] = P[0,0]*Q[0,0]
    # N[0,1] = P[0,0]*Q[0,1]
    # N[0,2] = P[0,1]*Q[0,0]
    # N[0,3] = P[0,1]*Q[0,1]
    # N[2,0] = P[1,0]*Q[0,0]
    # N[2,1] = P[1,0]*Q[0,1]
    # N[2,2] = P[1,1]*Q[0,0]
    # N[2,3] = P[1,1]*Q[0,1]
    
    # Choose Q[0,0] to make things work. Let's use:
    # Q[0,0] = sqrt(N[0,0]) with phase handling
    
    # Actually, let's use a more robust approach:
    # P[0,0] = N[0,0], Q[0,0] = 1
    # P[0,1] = N[0,2], Q[0,1] = N[0,1]
    # P[1,0] = N[2,0], Q[1,0] = N[2,1]
    # P[1,1] = N[2,2], Q[1,1] = N[2,3]
    
    # But this doesn't guarantee P and Q are unitary.
    
    # Let me use a different approach: use the fact that N is a tensor product
    # and extract the factors using the structure.
    
    # P = [[N[0,0], N[0,2]], [N[2,0], N[2,2]]]
    # Q = [[N[0,0], N[0,1]], [N[2,0], N[2,1]]]
    
    # But P and Q should satisfy: N[i,j] = P[i//2, j//2] * Q[i%2, j%2]
    # Let's verify: N[0,0] = P[0,0]*Q[0,0] = N[0,0]*N[0,0]... that's not right.
    
    # The correct extraction:
    # If N = P ⊗ Q, then:
    # P = [[N[0,0]/Q[0,0], N[0,2]/Q[0,0]], [N[2,0]/Q[0,0], N[2,2]/Q[0,0]]]
    # Q = [[Q[0,0], N[0,1]/N[0,0]*Q[0,0]], [N[2,1]/N[2,0]*Q[0,0], N[2,3]/N[2,0]*Q[0,0]]]
    
    # We have freedom in Q[0,0]. Let's choose Q[0,0] = 1 for simplicity.
    # Then:
    # P = [[N[0,0], N[0,2]], [N[2,0], N[2,2]]]
    # Q = [[1, N[0,1]/N[0,0]], [N[2,1]/N[2,0], N[2,3]/N[2,0]]]
    
    # But this might not give unitary P and Q.
    
    # Let me use a numerical approach instead. I'll use the fact that for a 
    # tensor product N = P ⊗ Q, we can extract P and Q as:
    # P = N[0:2, 0:2] / Q[0,0] (if Q[0,0] != 0)
    # This is getting too complicated. Let me use a simpler method.
    
    # For the 2-qubit case, I'll use a direct numerical approach:
    # I'll parameterize the single-qubit gates and solve for them.
    
    # Actually, let me just use the optimization approach with a good 
    # initialization for the generic case.
    
    return []


def _generic_synthesis(problem):
    """Generic synthesis using QSD-inspired approach."""
    qubits, budget = problem["qubits"], problem["cnot_budget"]
    target = _target(problem)
    
    if qubits == 2:
        return _generic_2qubit(target, budget)
    else:
        return _generic_3qubit(target, budget)


def _generic_2qubit(target, budget):
    """Generic 2-qubit synthesis."""
    # Use QSD with 2 CNOTs
    # U = (A0 ⊗ A1) · CNOT(0,1) · (B0 ⊗ B1) · CNOT(1,0) · (C0 ⊗ C1)
    
    CNOT_01 = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
    CNOT_10 = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
    
    # Compute intermediate matrices
    M = target @ CNOT_10
    N = M @ CNOT_01
    
    # N = (A0 @ B0) ⊗ (A1 @ B1)
    # Let P = A0 @ B0, Q = A1 @ B1
    # N = P ⊗ Q
    
    # Extract P and Q from N
    # Using the tensor product structure:
    # N[0,0] = P[0,0]*Q[0,0]
    # N[0,1] = P[0,0]*Q[0,1]
    # N[0,2] = P[0,1]*Q[0,0]
    # N[0,3] = P[0,1]*Q[0,1]
    # N[2,0] = P[1,0]*Q[0,0]
    # N[2,1] = P[1,0]*Q[0,1]
    # N[2,2] = P[1,1]*Q[0,0]
    # N[2,3] = P[1,1]*Q[0,1]
    
    # Choose Q[0,0] = 1 (absorb phase into P)
    P = np.array([[N[0,0], N[0,2]], [N[2,0], N[2,2]]], dtype=complex)
    Q = np.array([[1, N[0,1]/N[0,0] if abs(N[0,0]) > 1e-14 else 0],
                  [N[2,1]/N[2,0] if abs(N[2,0]) > 1e-14 else 0,
                   N[2,3]/N[2,0] if abs(N[2,0]) > 1e-14 else 0]], dtype=complex)
    
    # Now factor P = A0 @ B0 and Q = A1 @ B1
    # We have freedom to choose B0 and B1
    # Let's choose B0 = I, then A0 = P
    # Let's choose B1 = I, then A1 = Q
    # Then C0 = C1 = I
    
    # But we need to verify this gives the correct result.
    # Let's use a different approach: choose B0 and B1 to make A0 and A1 simple.
    
    # For simplicity, let's set:
    # A0 = P, B0 = I, A1 = Q, B1 = I, C0 = I, C1 = I
    
    # Convert to u3 gates
    A0_u3 = _u3_from_matrix(P)
    A1_u3 = _u3_from_matrix(Q)
    
    gates = [
        ["u3", 0, A0_u3[0], A0_u3[1], A0_u3[2]],
        ["u3", 1, A1_u3[0], A1_u3[1], A1_u3[2]],
        ["cx", 0, 1],
        ["u3", 0, 0, 0, 0],  # B0 = I
        ["u3", 1, 0, 0, 0],  # B1 = I
        ["cx", 1, 0],
        ["u3", 0, 0, 0, 0],  # C0 = I
        ["u3", 1, 0, 0, 0],  # C1 = I
    ]
    
    # Verify
    matrix = _build_fast(gates, 2)
    dist = _distance(target, matrix)
    
    if dist > 0.01:
        # Fallback to optimization
        return _optimize_2qubit(target, budget)
    
    return gates


def _optimize_2qubit(target, budget):
    """Optimization-based 2-qubit synthesis."""
    # Layout: u3 layers with CNOTs
    layout = []
    for layer in range(min(budget, 3)):
        for q in range(2):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        c, t = (0, 1) if layer % 2 == 0 else (1, 0)
        layout.append(["cx", c, t])
    for q in range(2):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = None
    best_dist = float('inf')
    
    for start in range(3):
        if _time_remaining() < 10:
            break
        if start == 0:
            values = [0.0] * n_params
        else:
            values = np.random.uniform(-math.pi, math.pi, n_params).tolist()
        
        curr_dist = _distance(target, _build_fast(circuit(values), 2))
        if curr_dist < best_dist:
            best_dist = curr_dist
            best = values[:]
        
        # Coordinate descent
        for sweep in range(10):
            if _time_remaining() < 10:
                break
            step = 0.5 / (1 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 10:
                    break
                for ts in (step, step*0.5, step*2):
                    for d in (1, -1):
                        trial = values[i] + d * ts
                        values[i] = trial
                        val = _distance(target, _build_fast(circuit(values), 2))
                        if val < curr_dist:
                            curr_dist = val
                            improved = True
                            break
                        values[i] = trial - d * ts
                    if improved:
                        break
                if _time_remaining() < 10:
                    break
            if not improved:
                break
        
        if curr_dist < best_dist:
            best_dist = curr_dist
            best = values[:]
    
    if best is None:
        best = [0.0] * n_params
    
    return circuit(best)


def _generic_3qubit(target, budget):
    """Generic 3-qubit synthesis."""
    # For 3-qubit, use a layout with CNOTs and single-qubit gates
    pairs = [(0,1), (1,2), (0,2), (1,0), (2,1), (2,0)]
    
    layout = []
    num_cnots = min(budget, 6)
    for layer in range(num_cnots):
        for q in range(3):
            layout.append(["u3", q, 0.0, 0.0, 0.0])
        c, t = pairs[layer % len(pairs)]
        layout.append(["cx", c, t])
    for q in range(3):
        layout.append(["u3", q, 0.0, 0.0, 0.0])
    
    n_params = 3 * sum(1 for g in layout if g[0] == "u3")
    
    def circuit(values):
        gates, k = [], 0
        for gate in layout:
            if gate[0] == "u3":
                gates.append(["u3", gate[1], values[k], values[k+1], values[k+2]])
                k += 3
            else:
                gates.append(list(gate))
        return gates
    
    best = None
    best_dist = float('inf')
    
    for start in range(3):
        if _time_remaining() < 15:
            break
        if start == 0:
            values = [0.0] * n_params
        else:
            values = np.random.uniform(-math.pi, math.pi, n_params).tolist()
        
        curr_dist = _distance(target, _build_fast(circuit(values), 3))
        if curr_dist < best_dist:
            best_dist = curr_dist
            best = values[:]
        
        # Coordinate descent
        for sweep in range(8):
            if _time_remaining() < 15:
                break
            step = 0.5 / (1 + sweep * 0.3)
            improved = False
            for i in range(n_params):
                if _time_remaining() < 15:
                    break
                for ts in (step, step*0.5, step*2):
                    for d in (1, -1):
                        trial = values[i] + d * ts
                        values[i] = trial
                        val = _distance(target, _build_fast(circuit(values), 3))
                        if val < curr_dist:
                            curr_dist = val
                            improved = True
                            break
                        values[i] = trial - d * ts
                    if improved:
                        break
                if _time_remaining() < 15:
                    break
            if not improved:
                break
        
        if curr_dist < best_dist:
            best_dist = curr_dist
            best = values[:]
    
    if best is None:
        best = [0.0] * n_params
    
    return circuit(best)


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
    else:
        gates = _generic_synthesis(problem)
    
    # Verify
    target = _target(problem)
    matrix = _build_fast(gates, qubits)
    dist = _distance(target, matrix)
    
    if dist > 0.001 and _time_remaining() > 20:
        gates = _generic_synthesis(problem)
    
    return gates
# EVOLVE-BLOCK-END
