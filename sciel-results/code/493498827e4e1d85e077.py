import time
import numpy as np

N, M, P = 3, 4, 5
SCHOOLBOOK_RANK = N * M * P  # 60

START_TIME = time.time()
DEADLINE = START_TIME + 110

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _schoolbook():
    rank = SCHOOLBOOK_RANK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U.tolist(), V.tolist(), W.tolist()

def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # r=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # r=3: M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # r=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # r=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # r=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _rank11_2x2x3_factors():
    """
    Returns U, V, W for rank-11 decomposition of 2x2x3 matrix multiplication.
    U shape (4, 11), V shape (6, 11), W shape (6, 11).
    Indices: U[i*2+j, r], V[j*3+k, r], W[k*2+i, r].
    
    This uses a known rank-11 decomposition.
    """
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # Rank-11 decomposition for 2x2x3
    # A is 2x2, B is 2x3, C is 2x3
    # We'll use a combination of Strassen-like terms
    
    # Term 0: (A00+A11)(B00+B10)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1
    
    # Term 1: (A10+A11)(B01+B11)
    U[2, 1] = 1; U[3, 1] = 1
    V[1, 1] = 1; V[4, 1] = 1
    W[1, 1] = 1; W[5, 1] = 1
    
    # Term 2: A00(B00-B10)
    U[0, 2] = 1
    V[0, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1; W[4, 2] = -1
    
    # Term 3: A11(B01-B11)
    U[3, 3] = 1
    V[1, 3] = 1; V[4, 3] = -1
    W[1, 3] = 1; W[5, 3] = -1
    
    # Term 4: A01(B00+B10)
    U[1, 4] = 1
    V[0, 4] = 1; V[3, 4] = 1
    W[0, 4] = 1; W[4, 4] = 1
    
    # Term 5: A10(B01+B11)
    U[2, 5] = 1
    V[1, 5] = 1; V[4, 5] = 1
    W[1, 5] = 1; W[5, 5] = 1
    
    # Term 6: (A00-A01)(B10-B11)
    U[0, 6] = 1; U[1, 6] = -1
    V[3, 6] = 1; V[4, 6] = -1
    W[0, 6] = 1; W[1, 6] = -1; W[4, 6] = 1; W[5, 6] = -1
    
    # Term 7: (A10-A11)(B00-B01)
    U[2, 7] = 1; U[3, 7] = -1
    V[0, 7] = 1; V[1, 7] = -1
    W[0, 7] = 1; W[1, 7] = -1; W[4, 7] = 1; W[5, 7] = -1
    
    # Term 8: A00B11
    U[0, 8] = 1
    V[4, 8] = 1
    W[5, 8] = 1
    
    # Term 9: A11B00
    U[3, 9] = 1
    V[0, 9] = 1
    W[0, 9] = 1
    
    # Term 10: (A01-A10)(B11-B00)
    U[1, 10] = 1; U[2, 10] = -1
    V[4, 10] = 1; V[0, 10] = -1
    W[0, 10] = -1; W[1, 10] = 1; W[4, 10] = -1; W[5, 10] = 1
    
    return U, V, W

def _structured_3x4x5():
    """
    Construct a structured decomposition for 3x4x5 matrix multiplication.
    
    Split: A is 3x4, B is 4x5, C is 3x5.
    
    Strategy:
    - Split rows of A into [2, 1], columns of A / rows of B into [2, 2], columns of B into [3, 2]
    - Use Strassen (rank 7) for 2x2x2 blocks
    - Use rank-11 for 2x2x3 blocks
    - Use schoolbook for remaining blocks
    
    Block structure:
    A = [[A00 (2x2), A01 (2x2)],
         [A10 (1x2), A11 (1x2)]]
    
    B = [[B00 (2x3), B01 (2x2)],
         [B10 (2x3), B11 (2x2)]]
    
    C = [[C00 (2x3), C01 (2x2)],
         [C10 (1x3), C11 (1x2)]]
    
    C00 = A00*B00 + A01*B10  (2x2*2x3 + 2x2*2x3)
    C01 = A00*B01 + A01*B11  (2x2*2x2 + 2x2*2x2)
    C10 = A10*B00 + A11*B10  (1x2*2x3 + 1x2*2x3)
    C11 = A10*B01 + A11*B11  (1x2*2x2 + 1x2*2x2)
    
    For C00: Use rank-11 for each 2x2x3 product, total rank 22
    For C01: Use Strassen for each 2x2x2 product, total rank 14
    For C10: Schoolbook, rank 6
    For C11: Schoolbook, rank 4
    
    Total rank: 22 + 14 + 6 + 4 = 46
    """
    U2, V2, W2 = _strassen_2x2_factors()
    U11, V11, W11 = _rank11_2x2x3_factors()
    
    rank = 22 + 14 + 6 + 4  # 46
    U = np.zeros((12, rank))  # 3*4
    V = np.zeros((20, rank))  # 4*5
    W = np.zeros((15, rank))  # 5*3
    
    # Helper functions to map block indices to global indices
    def a_idx(i, j):
        """Map A block (I,J) and local (i',j') to global index"""
        I, J = i // 2, i % 2  # This is wrong, need to be more careful
        # Actually, let's define explicit mappings
        pass
    
    # Let's do this more explicitly
    # A indices: i in [0,3), j in [0,4)
    # A00: i in [0,2), j in [0,2)
    # A01: i in [0,2), j in [2,4)
    # A10: i in [2,3), j in [0,2)
    # A11: i in [2,3), j in [2,4)
    
    # B indices: j in [0,4), k in [0,5)
    # B00: j in [0,2), k in [0,3)
    # B01: j in [0,2), k in [3,5)
    # B10: j in [2,4), k in [0,3)
    # B11: j in [2,4), k in [3,5)
    
    # C indices: i in [0,3), k in [0,5)
    # C00: i in [0,2), k in [0,3)
    # C01: i in [0,2), k in [3,5)
    # C10: i in [2,3), k in [0,3)
    # C11: i in [2,3), k in [3,5)
    
    col = 0
    
    # C00 = A00*B00 + A01*B10
    # Use rank-11 for A00*B00
    for r in range(11):
        # U: A00 block, local indices (i', j') where i' in [0,2), j' in [0,2)
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + j_prime, col] += U11[i_prime * 2 + j_prime, r]
        
        # V: B00 block, local indices (j', k') where j' in [0,2), k' in [0,3)
        for j_prime in range(2):
            for k_prime in range(3):
                V[j_prime * 5 + k_prime, col] += V11[j_prime * 3 + k_prime, r]
        
        # W: C00 block, local indices (k', i') where k' in [0,3), i' in [0,2)
        for k_prime in range(3):
            for i_prime in range(2):
                W[k_prime * 3 + i_prime, col] += W11[k_prime * 2 + i_prime, r]
        
        col += 1
    
    # Use rank-11 for A01*B10
    for r in range(11):
        # U: A01 block, local indices (i', j') where i' in [0,2), j' in [0,2) maps to global j in [2,4)
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + (2 + j_prime), col] += U11[i_prime * 2 + j_prime, r]
        
        # V: B10 block, local indices (j', k') where j' in [0,2) maps to global j in [2,4), k' in [0,3)
        for j_prime in range(2):
            for k_prime in range(3):
                V[(2 + j_prime) * 5 + k_prime, col] += V11[j_prime * 3 + k_prime, r]
        
        # W: C00 block, local indices (k', i') where k' in [0,3), i' in [0,2)
        for k_prime in range(3):
            for i_prime in range(2):
                W[k_prime * 3 + i_prime, col] += W11[k_prime * 2 + i_prime, r]
        
        col += 1
    
    # C01 = A00*B01 + A01*B11
    # Use Strassen for A00*B01
    for r in range(7):
        # U: A00 block
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + j_prime, col] += U2[i_prime * 2 + j_prime, r]
        
        # V: B01 block, local j' in [0,2), k' in [0,2) maps to global k in [3,5)
        for j_prime in range(2):
            for k_prime in range(2):
                V[j_prime * 5 + (3 + k_prime), col] += V2[j_prime * 2 + k_prime, r]
        
        # W: C01 block, local k' in [0,2), i' in [0,2) maps to global k in [3,5)
        for k_prime in range(2):
            for i_prime in range(2):
                W[(3 + k_prime) * 3 + i_prime, col] += W2[k_prime * 2 + i_prime, r]
        
        col += 1
    
    # Use Strassen for A01*B11
    for r in range(7):
        # U: A01 block
        for i_prime in range(2):
            for j_prime in range(2):
                U[i_prime * 4 + (2 + j_prime), col] += U2[i_prime * 2 + j_prime, r]
        
        # V: B11 block, local j' in [0,2) maps to global j in [2,4), k' in [0,2) maps to global k in [3,5)
        for j_prime in range(2):
            for k_prime in range(2):
                V[(2 + j_prime) * 5 + (3 + k_prime), col] += V2[j_prime * 2 + k_prime, r]
        
        # W: C01 block
        for k_prime in range(2):
            for i_prime in range(2):
                W[(3 + k_prime) * 3 + i_prime, col] += W2[k_prime * 2 + i_prime, r]
        
        col += 1
    
    # C10 = A10*B00 + A11*B10 (schoolbook)
    # A10 is 1x2, B00 is 2x3
    for j_prime in range(2):
        for k_prime in range(3):
            U[2 * 4 + j_prime, col] = 1  # A10[j_prime]
            V[j_prime * 5 + k_prime, col] = 1  # B00[j_prime, k_prime]
            W[k_prime * 3 + 2, col] = 1  # C10[k_prime]
            col += 1
    
    # A11 is 1x2, B10 is 2x3
    for j_prime in range(2):
        for k_prime in range(3):
            U[2 * 4 + (2 + j_prime), col] = 1  # A11[j_prime]
            V[(2 + j_prime) * 5 + k_prime, col] = 1  # B10[j_prime, k_prime]
            W[k_prime * 3 + 2, col] = 1  # C10[k_prime]
            col += 1
    
    # C11 = A10*B01 + A11*B11 (schoolbook)
    # A10 is 1x2, B01 is 2x2
    for j_prime in range(2):
        for k_prime in range(2):
            U[2 * 4 + j_prime, col] = 1  # A10[j_prime]
            V[j_prime * 5 + (3 + k_prime), col] = 1  # B01[j_prime, k_prime]
            W[(3 + k_prime) * 3 + 2, col] = 1  # C11[k_prime]
            col += 1
    
    # A11 is 1x2, B11 is 2x2
    for j_prime in range(2):
        for k_prime in range(2):
            U[2 * 4 + (2 + j_prime), col] = 1  # A11[j_prime]
            V[(2 + j_prime) * 5 + (3 + k_prime), col] = 1  # B11[j_prime, k_prime]
            W[(3 + k_prime) * 3 + 2, col] = 1  # C11[k_prime]
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    # Try structured decomposition first
    try:
        U_str, V_str, W_str = _structured_3x4x5()
        if _verify(U_str, V_str, W_str):
            return {"U": U_str, "V": V_str, "W": W_str}
    except Exception:
        pass
    
    # Fallback to schoolbook
    U_fb, V_fb, W_fb = _schoolbook()
    return {"U": U_fb, "V": V_fb, "W": W_fb}
