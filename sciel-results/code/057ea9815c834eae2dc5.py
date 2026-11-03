"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    # Outer Strassen coefficients
    outer_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]
    
    outer_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]
    
    outer_C = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11(-1)
        [0, 1, 0, 1],   # M3 -> C01, C11
        [-1, 0, 1, 0],  # M4 -> C00(-1), C10
        [1, 1, 0, 0],   # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
    ]
    
    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]
                    
                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W
    
    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    U: (n*m, n*m*p), V: (m*p, n*m*p), W: (p*n, n*m*p)
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x8():
    """
    Decomposition of M_{4,4,8} using Kronecker product of M_{2,2,2} (Strassen, rank 7)
    with M_{2,2,2} (Strassen, rank 7). Total rank = 49.
    
    The idea: 4x4x8 = (2x2x2) tensor product (2x2x2) where the second factor covers
    a 2x2 block in k-space. Wait, that gives 4x4x4, not 4x4x8.
    
    Actually, let me think differently. We can split the 9 columns into 8+1.
    For the 8 columns, we can use a different decomposition.
    
    Actually, let me use a simpler approach: split 4x4x9 into 4x4x4 + 4x4x4 + 4x4x1.
    Rank = 49 + 49 + 16 = 114. Same as before.
    
    Let me try to be smarter. Use Strassen for 4x4x4 twice (covering 8 k-values) 
    plus schoolbook for 4x4x1.
    """
    n, m, p = 4, 4, 8
    U444, V444, W444 = _decomp_4x4x4()
    U1, V1, W1 = U444.copy(), V444.copy(), W444.copy()
    U2, V2, W2 = U444.copy(), V444.copy(), W444.copy()
    
    total_rank = 49 + 49
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # First 4x4x4 block covers k=0..3
    U[:, :49] = U1
    for j in range(m):
        for k in range(4):
            V[j * 8 + k, :49] = V1[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W1[k * 4 + i, :]
    
    # Second 4x4x4 block covers k=4..7
    U[:, 49:] = U2
    for j in range(m):
        for k in range(4):
            V[j * 8 + (4 + k), 49:] = V2[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * 4 + i, 49:] = W2[k * 4 + i, :]
    
    return U, V, W

def _decomp_4x4x9_kron():
    """
    Decomposition of M_{4,4,9} using a Kronecker-product-style approach.
    
    Strategy: Use Strassen's 2x2x2 decomposition (rank 7) applied at the block level.
    We view the 4x4x9 as a block structure and apply tensor product decompositions.
    
    Alternative: Direct construction using the fact that 4x4x9 = (2x2) x (2x4) in some sense.
    
    Let me try: split into 4x4x8 + 4x4x1.
    For 4x4x8: use two 4x4x4 Strassen blocks = rank 98.
    For 4x4x1: schoolbook = rank 16.
    Total: 114.
    
    Can we do better? Let's try a mixed approach.
    Use Strassen for 2x2x2 blocks where possible.
    
    Actually, let me try: 4x4x9 as a sum of rank-1 terms + Strassen blocks.
    
    For the first 8 columns (k=0..7), use 4 Strassen 2x2x2 blocks (each covering 
    a 2x2x2 sub-block). That's 4 * 7 = 28 rank terms.
    For the last column (k=8), use 16 rank-1 terms (schoolbook 4x4x1).
    Total: 28 + 16 = 44.
    
    Wait, that doesn't work because the Strassen blocks need to cover all i, j dimensions.
    Each 2x2x2 Strassen block covers 2 i's, 2 j's, and 2 k's.
    So for 4x4x8, we need 2*2*2 = 8 such blocks, each of rank 7. Total = 56.
    Plus 16 for the last column. Total = 72.
    
    Hmm, that's worse than 114? No, 72 < 114. Let me verify this construction.
    """
    n, m, p = 4, 4, 9
    # 8 blocks of 2x2x2 (covering k=0..7) + schoolbook 4x4x1 (k=8)
    rank_strassen_blocks = 8 * 7  # 56
    rank_schoolbook = 16
    total_rank = rank_strassen_blocks + rank_schoolbook  # 72
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    inner_U, inner_V, inner_W = _strassen_2x2()
    
    col = 0
    # 8 blocks of 2x2x2 for k=0..7
    for bi in range(2):  # i-block: 0 or 1 (covers rows 2*bi, 2*bi+1)
        for bj in range(2):  # j-block: 0 or 1 (covers cols 2*bj, 2*bj+1)
            for bk in range(4):  # k-block: 0,1,2,3 (covers k=2*bk, 2*bk+1)
                for r in range(7):
                    # For each inner rank term r
                    for li in range(2):
                        for lj in range(2):
                            a_idx = 2 * bi + li  # global i
                            b_idx = 2 * bj + lj  # global j
                            u_val = inner_U[li * 2 + lj, r]
                            U[a_idx * m + b_idx, col] = u_val
                    
                    for lj in range(2):
                        for lk in range(2):
                            b_idx = 2 * bj + lj  # global j
                            k_idx = 2 * bk + lk  # global k
                            v_val = inner_V[lj * 2 + lk, r]
                            V[b_idx * p + k_idx, col] = v_val
                    
                    for lk in range(2):
                        for li in range(2):
                            k_idx = 2 * bk + lk  # global k
                            a_idx = 2 * bi + li  # global i
                            w_val = inner_W[lk * 2 + li, r]
                            W[k_idx * n + a_idx, col] = w_val
                    
                    col += 1
    
    # Schoolbook 4x4x1 for k=8
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 8, col] = 1
            W[8 * n + i, col] = 1
            col += 1
    
    return U, V, W

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    # Sum over rank: for each (i,j,k), sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] should be 1
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    
    return np.allclose(result, 1.0, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the Kronecker product approach: rank 72
    U, V, W = _decomp_4x4x9_kron()
    
    if time.time() < deadline - 10:
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: previous approach (rank 114)
    U444, V444, W444 = _decomp_4x4x4()
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 * 2 + rank_441  # 114
    
    U2 = np.zeros((n * m, total_rank))
    V2 = np.zeros((m * p, total_rank))
    W2 = np.zeros((p * n, total_rank))
    
    # First 4x4x4 block covers k=0..3
    U2[:, :49] = U444
    for j in range(m):
        for k in range(4):
            V2[j * 9 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W2[k * 4 + i, :49] = W444[k * 4 + i, :]
    
    # Second 4x4x4 block covers k=4..7
    U2[:, 49:98] = U444
    for j in range(m):
        for k in range(4):
            V2[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W2[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]
    
    # Schoolbook 4x4x1 for k=8
    for i in range(n):
        for j in range(m):
            U2[i * m + j, 98 + i * m + j] = 1
            V2[j * p + 8, 98 + i * m + j] = 1
            W2[8 * n + i, 98 + i * m + j] = 1
    
    if time.time() < deadline - 5:
        if _verify_decomp(U2, V2, W2, n, m, p):
            return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    
    # Final fallback: schoolbook (rank 144)
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END

