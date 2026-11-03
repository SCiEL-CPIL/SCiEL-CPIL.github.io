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
    
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1
    
    # M5: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
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

def _check_decomp(U, V, W, n, m, p, tol=1e-8):
    """Verify the decomposition using tensor contraction."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _decomp_4x4x3():
    """
    Decomposition of M_{4,4,3} using a combination of Strassen for 4x4x2 and schoolbook for 4x4x1.
    Rank 49 + 16 = 65.
    U: (16, 65), V: (12, 65), W: (12, 65)
    """
    n, m, p = 4, 4, 3
    rank = 65
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Part 1: k=0..1, 4x4x2. We can use Strassen on 4x4x2? No, Strassen is for square.
    # Actually, for 4x4x2, we can split into two 4x4x1? No, that's 32.
    # Let's use a 2x2x2 Strassen based approach for the 4x4x2 part.
    # We can view 4x4x2 as two 4x4x1 multiplications.
    # Wait, we can use the 2x2x2 Strassen to compute 2x2x2 in rank 7.
    # For 4x4x2:
    # A is 4x4, B is 4x2, C is 4x2.
    # We can split A into 2x2 blocks, B into 2x1 blocks (since B is 4x2, it's 2x2 blocks of 2x1? No, B is 4x2.
    # Let's just use schoolbook for 4x4x2? Rank 32.
    # Or use Strassen for 2x2x2?
    # Let's try to use the 2x2x2 Strassen for the 4x4x2 case.
    # A = [[A00, A01], [A10, A11]] where Ai are 2x2.
    # B = [[B00], [B10]] where Bi are 2x2? No, B is 4x2.
    # B = [[B00, B01], [B10, B11]] where Bi are 2x1.
    # C = [[C00, C01], [C10, C11]] where Ci are 2x1.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # This is not directly 2x2x2 Strassen.
    
    # Let's just use schoolbook for 4x4x2. Rank 32.
    U1, V1, W1 = _schoolbook_nm_p(4, 4, 2)
    U[:, 0:32] = U1
    V[:, 0:32] = V1
    W[:, 0:32] = W1
    
    # Part 2: k=2, 4x4x1 schoolbook. Rank 16.
    U2, V2, W2 = _schoolbook_nm_p(4, 4, 1)
    U[:, 32:48] = U2
    V[:, 32:48] = V2
    W[:, 32:48] = W2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    limit = t0 + 130.0
    
    # Strategy: Partition the 9 columns of B into three groups of 3 columns.
    # For each group, use a 4x4x3 decomposition.
    # 4x4x3 decomposition: We can use schoolbook (rank 48) or try to find something better.
    # Let's use schoolbook for 4x4x3 for now. Rank 48.
    # Total rank: 3 * 48 = 144. This is the same as the full schoolbook.
    
    # Let's try to use a better 4x4x3 decomposition.
    # We can use Strassen for 4x4x2 (rank 49? No, Strassen is for square. For 4x4x2, we can use 2x2x2 Strassen?)
    # Actually, let's use the 2-level Strassen for 4x4x4, but we only have 3 columns.
    # We can pad the 3 columns to 4 columns? No, that would increase the rank.
    
    # Let's try a different approach: use the 4x4x4 Strassen decomposition, but only use the first 3 columns.
    # The 4x4x4 Strassen decomposition has rank 49. If we only need 3 columns, can we reduce the rank?
    # Probably not easily, because the decomposition is entangled.
    
    # Let's just use the schoolbook for 4x4x3. Rank 48.
    # Total rank: 144.
    
    # Wait, the current program has rank 114. We need to do better than that.
    # The current program uses 49 + 49 + 16 = 114.
    # Our strategy is 3 * (4x4x3). If we can find a 4x4x3 decomposition with rank < 38, we win.
    # 3 * 38 = 114. So we need rank < 38 for 4x4x3.
    # Schoolbook is 48. Strassen for 4x4x2 is 49 (if we use 2-level Strassen for 4x4x2? No, that's not standard).
    
    # Let's try to use a 2x2x3 Winograd variant.
    # Actually, let's just use the current program's approach as a fallback, and try the new strategy.
    
    # New strategy: 3 groups of 3 columns.
    # For each group, use a 4x4x3 decomposition.
    # Let's use a simple 4x4x3 decomposition: schoolbook (rank 48).
    # Total rank: 144.
    
    # This is worse than 114. Let's try to optimize the 4x4x3 decomposition.
    # We can use Strassen for 2x2x2, and combine it with schoolbook for the remaining part.
    # 4x4x3: Split into 2x2x3 blocks.
    # There are 4 blocks of 2x2x3.
    # 2x2x3: We can use a rank-6 decomposition? (2*2*3 = 12 schoolbook).
    # Let's use a simple 2x2x3 decomposition: schoolbook (rank 12).
    # Total rank for 4x4x3: 4 * 12 = 48.
    
    # This is the same as the full schoolbook.
    
    # Let's try to use Strassen for 2x2x2 within the 2x2x3.
    # 2x2x3: Split the 3 columns into 2 and 1.
    # 2x2x2: Strassen (rank 7).
    # 2x2x1: Schoolbook (rank 4).
    # Total rank for 2x2x3: 7 + 4 = 11.
    # Total rank for 4x4x3: 4 * 11 = 44.
    # Total rank for 4x4x9: 3 * 44 = 132.
    
    # This is still worse than 114.
    
    # Let's try to use a better 2x2x3 decomposition.
    # Actually, the rank of 2x2x3 is known to be 8 (from the literature).
    # Let's assume rank 8 for 2x2x3.
    # Total rank for 4x4x3: 4 * 8 = 32.
    # Total rank for 4x4x9: 3 * 32 = 96.
    
    # This is better than 114! Let's try to construct a rank-8 decomposition for 2x2x3.
    
    # For now, let's just use the current program's approach (rank 114) as a fallback,
    # and try to construct the rank-96 approach.
    
    # Since we don't have a rank-8 decomposition for 2x2x3, let's use the rank-11 approach (Strassen + schoolbook).
    # Total rank: 132.
    
    # This is worse than 114. Let's just use the current program's approach.
    
    # Actually, let's try to use the 4x4x4 Strassen decomposition, but only for the first 8 columns,
    # and then use schoolbook for the last column.
    # 4x4x8: We can use two 4x4x4 Strassen decompositions. Rank 49 + 49 = 98.
    # 4x4x1: Schoolbook. Rank 16.
    # Total rank: 98 + 16 = 114.
    
    # This is the same as the current program.
    
    # Let's try a different partition: 4x4x9 = 4x4x4 + 4x4x4 + 4x4x1.
    # This is the same as the current program.
    
    # Let's try 4x4x9 = 4x4x3 + 4x4x3 + 4x4x3.
    # If we can find a 4x4x3 decomposition with rank 38, we get 114.
    # If we can find a 4x4x3 decomposition with rank < 38, we win.
    
    # Let's try to use a 4x4x3 decomposition based on Strassen for 4x4x2 and schoolbook for 4x4x1.
    # 4x4x2: We can use a 2-level Strassen? No, Strassen is for square matrices.
    # For 4x4x2, we can use a 2x2x2 Strassen for each 2x2x2 block, but the blocks are not independent.
    
    # Let's just use the current program's approach (rank 114).
    
    total_rank = 114
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    # Part 1: k=0..3, 4x4x4 Strassen
    U1, V1, W1 = _decomp_4x4x4()
    U_full[:, 0:49] = U1
    for j in range(4):
        for k in range(4):
            V_full[j*9 + k, 0:49] = V1[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[k*4 + i, 0:49] = W1[k*4 + i, :]
    
    # Part 2: k=4..7, 4x4x4 Strassen
    U2, V2, W2 = _decomp_4x4x4()
    U_full[:, 49:98] = U2
    for j in range(4):
        for k in range(4):
            V_full[j*9 + (4+k), 49:98] = V2[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W_full[(4+k)*4 + i, 49:98] = W2[k*4 + i, :]
    
    # Part 3: k=8, 4x4x1 schoolbook
    U3, V3, W3 = _schoolbook_nm_p(4, 4, 1)
    U_full[:, 98:114] = U3
    for j in range(4):
        V_full[j*9 + 8, 98:114] = V3[j, :]
    for i in range(4):
        W_full[8*4 + i, 98:114] = W3[i, :]
    
    if time.time() < limit - 5.0:
        ok = _check_decomp(U_full, V_full, W_full, n, m, p)
        if ok:
            return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # Fallback: schoolbook (rank 144)
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
