"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_decomposition_v2():
    """Build a decomposition for 4x4 x 4x8 using a recursive block approach.
    Use Strassen for the inner 2x2 blocks within each 4x4 matrix product.
    Structure: 4x4 * 4x8. We can think of this as 8 separate 4x4*4x1 multiplications.
    For each 4x4*4x1, we can use a 4x4 algorithm.
    
    Actually, a better structural approach: 
    The tensor M_{4,4,8} has structure (i,j,k) where i in 0..3, j in 0..3, k in 0..7.
    We can group k into two blocks of 4: k = 4*l + k' where l in {0,1}, k' in 0..3.
    Then M_{i,j,4l+k'} = sum_c U[i,j,c] * V[j,4l+k',c] * W[4l+k',i,c]
    
    This is equivalent to 2 independent 4x4*4x4 multiplications, each with rank 49 via Strassen.
    Total rank: 98.
    
    But can we do better by exploiting the fact that the A matrix is shared?
    Let's try a different decomposition:
    Write A as 2x2 blocks of 2x2 matrices: A = [[A00, A01], [A10, A11]] where each is 2x2.
    B as 2x4 blocks: B = [[B00, B01], [B10, B11]] where B00 is 2x4, etc.
    C = [[C00, C01], [C10, C11]] where each is 2x4.
    
    C_ij = A_ij * B_jk (summing over j).
    For the 2x2 block structure with 2x4 B blocks, we need a generalized Strassen.
    
    Alternatively, use the fact that 4x4 * 4x8 can be seen as 4x4 * (4x4) * I_2 (block identity).
    So we can use a 4x4*4x4 algorithm and replicate it twice.
    Rank: 2 * 49 = 98.
    
    Let's try to use the Winograd algorithm or a different factorization.
    For 4x4 matrix multiplication, the best known rank is 49 (Strassen's algorithm applied recursively).
    So 2 * 49 = 98 is the natural bound.
    
    But wait - can we do better by not using Strassen's algorithm for the full 4x4, but instead
    using a more efficient algorithm for 4x4 * 4x8 directly?
    
    Actually, let's try a different approach: use the fact that we can factor the computation
    differently. The tensor M_{4,4,8} = M_{4,4,4} ⊗ M_{4,4,2} where the second factor
    is a "identity" tensor (since 4x4 * 4x2 with the B matrix being 4x2, we just need to
    multiply each column of A by the corresponding column of B).
    
    Wait, that's not quite right. Let's think of it as:
    C = A * B where A is 4x4 and B is 4x8.
    We can write B = [B_0, B_1] where B_0 and B_1 are both 4x4.
    Then C = [A*B_0, A*B_1].
    So we need two independent 4x4*4x4 multiplications.
    Each requires rank 49 (using Strassen), so total rank is 98.
    
    This is the same as before. Let's just implement it cleanly.
    """
    n, m, p = 4, 4, 8
    
    # Build the Strassen 2x2*2x2*2x2 decomposition
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[2, 0] = 1; W2[1, 0] = 1; W2[3, 0] = -1
    
    # M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Build 4x4*4x4*4x4 using Kronecker product of Strassen
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            U4[u_idx, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            v_idx = (j1 * 2 + j2) * 4 + (k1 * 2 + k2)
                            V4[v_idx, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            w_idx = (k1 * 2 + k2) * 4 + (i1 * 2 + i2)
                            W4[w_idx, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
    
    # Now build the 4x4 * 4x8 decomposition
    # Split into two 4x4 * 4x4 problems
    r = 98
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # First 4x4 * 4x4: columns 0-47
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, 0:49]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, 0:49] = W4[k * 4 + i, 0:49]
    
    # Second 4x4 * 4x4: columns 49-97
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j * 8 + k, 49:98] = V4[j * 4 + (k - 4), 0:49]
    for k in range(4, 8):
        for i in range(4):
            W[k * 4 + i, 49:98] = W4[(k - 4) * 4 + i, 0:49]
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-10:
                    # Fallback to schoolbook
                    rank_fb = n * m * p
                    U_fb = np.zeros((n * m, rank_fb))
                    V_fb = np.zeros((m * p, rank_fb))
                    W_fb = np.zeros((p * n, rank_fb))
                    col = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U_fb[ii * m + jj, col] = 1
                                V_fb[jj * p + kk, col] = 1
                                W_fb[kk * n + ii, col] = 1
                                col += 1
                    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    try:
        result = _build_decomposition_v2()
        if time.time() < deadline:
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
