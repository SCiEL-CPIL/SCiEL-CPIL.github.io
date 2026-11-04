"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return Strassen decomposition for 2x2 * 2x2 -> 2x2. Rank 7.
    Index convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c] with n=m=p=2.
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _strassen_4x4_4x4():
    """Strassen-based decomposition for 4x4 * 4x4 -> 4x4. Rank 49.
    Split into 2x2 blocks of size 2x2, use Strassen at block level,
    and Strassen (rank 7) for each 2x2*2x2 sub-product.
    """
    n = m = p = 4
    total_rank = 49
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_base, V_base, W_base = _strassen_2x2()
    
    def A_idx(bi, bj, di, ej):
        return (bi*2 + di) * 4 + (bj*2 + ej)
    
    def B_idx(bi, bj, di, ej):
        return (bi*2 + di) * 4 + (bj*2 + ej)
    
    def W_idx(bi, bj, di, ej):
        return (bj*2 + ej) * 4 + (bi*2 + di)
    
    strassen_steps2 = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),   # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),             # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),            # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),            # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),             # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),   # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),   # M7
    ]
    
    c_contribs = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }
    
    for s in range(7):
        a_terms, b_terms = strassen_steps2[s]
        col_offset = s * 7
        
        for c_inner in range(7):
            col = col_offset + c_inner
            
            for a_bi, a_bj, a_coeff in a_terms:
                for di in range(2):
                    for ej in range(2):
                        U[A_idx(a_bi, a_bj, di, ej), col] += a_coeff * U_base[di*2+ej, c_inner]
            
            for b_bi, b_bj, b_coeff in b_terms:
                for di in range(2):
                    for ej in range(2):
                        V[B_idx(b_bi, b_bj, di, ej), col] += b_coeff * V_base[di*2+ej, c_inner]
            
            for (c_bi, c_bj), contribs in c_contribs.items():
                for (step, c_coeff) in contribs:
                    if step == s:
                        for di in range(2):
                            for ej in range(2):
                                W[W_idx(c_bi, c_bj, di, ej), col] += c_coeff * W_base[di*2+ej, c_inner]
    
    return U, V, W


def _strassen_2x2_2x3():
    """Strassen-like decomposition for 2x2 * 2x3 -> 2x3. Rank 10.
    
    Split B into B_left (2x2) + B_right (2x1).
    Use Strassen for the 2x2*2x2 part (rank 7) and schoolbook for the 2x2*2x1 part (rank 4).
    But we can share: the 2x2*2x1 part only needs A * B_col, which is just 2 scalar mults per output row.
    
    Actually, let's think more carefully.
    A is 2x2, B is 2x3. C = A*B is 2x3.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    Schoolbook rank = 2*2*3 = 12.
    
    Strassen for 2x2*2x2 gives rank 7 for the first two columns.
    For the third column, we need A * B[:,2] which is a 2x2 * 2x1 product, rank 4 schoolbook.
    Total: 7 + 4 = 11.
    
    Can we do better? Let's try a different split.
    Actually, let's just use rank 11 (Strassen 7 + schoolbook 4 for the last column).
    """
    n = 2; m = 2; p = 3
    rank = 11
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2*2x2 for columns 0,1
    U_s, V_s, W_s = _strassen_2x2()
    U[:, :7] = U_s
    V[:, :7] = V_s  # V_s is 4x7, corresponds to B rows 0-1, cols 0-1
    # W_s is 4x7, W_s[k*2+i, c] where k is col of C (0,1), i is row of C
    W[:, :7] = W_s
    
    # Schoolbook for column 2: A (2x2) * B[:,2] (2x1) -> C[:,2] (2x1)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # 4 terms:
    col = 7
    # A[0,0]*B[0,2] -> C[0,2]
    U[0, col] = 1; V[2, col] = 1; W[4, col] = 1; col += 1  # W index: k=2, i=0 -> 2*2+0=4
    # A[0,1]*B[1,2] -> C[0,2]
    U[1, col] = 1; V[3, col] = 1; W[4, col] = 1; col += 1
    # A[1,0]*B[0,2] -> C[1,2]
    U[2, col] = 1; V[2, col] = 1; W[5, col] = 1; col += 1  # W index: k=2, i=1 -> 2*2+1=5
    # A[1,1]*B[1,2] -> C[1,2]
    U[3, col] = 1; V[3, col] = 1; W[5, col] = 1; col += 1
    
    return U, V, W


def _strassen_4x4_4x3():
    """Strassen-based decomposition for 4x4 * 4x3 -> 4x3. Rank 44.
    
    Split A into 2x2 blocks, B into 2x3 blocks (top/bottom).
    C_top (2x3) = A11*B1 + A12*B2
    C_bot (2x3) = A21*B1 + A22*B2
    
    Use Strassen-like sharing at the block level:
    M1 = (A11+A22)*B1  (2x2 * 2x3, rank 11)
    M2 = (A12+A22)*B2  (2x2 * 2x3, rank 11)
    M3 = A11*B2         (2x2 * 2x3, rank 11)
    M4 = A22*B1         (2x2 * 2x3, rank 11)
    
    C_top = M1 - M4 + M3
    C_bot = M1 - M2 + M4
    
    Total rank = 4 * 11 = 44.
    """
    n = 4; m = 4; p = 3
    total_rank = 44
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    U_base, V_base, W_base = _strassen_2x2_2x3()
    
    # A block indices:
    # A11 = rows 0-1, cols 0-1: U indices 0,1,4,5
    # A12 = rows 0-1, cols 2-3: U indices 2,3,6,7
    # A21 = rows 2-3, cols 0-1: U indices 8,9,12,13
    # A22 = rows 2-3, cols 2-3: U indices 10,11,14,15
    
    # B block indices (B is 4x3):
    # B1 = rows 0-1, cols 0-2: V indices 0,1,2,3,4,5
    # B2 = rows 2-3, cols 0-2: V indices 6,7,8,9,10,11
    
    # C block indices (C is 4x3, W index = k*4+i):
    # C_top = rows 0-1, cols 0-2: W indices for i=0,1 and k=0,1,2
    #   i=0: k=0->0, k=1->4, k=2->8
    #   i=1: k=0->1, k=1->5, k=2->9
    # C_bot = rows 2-3, cols 0-2: W indices for i=2,3 and k=0,1,2
    #   i=2: k=0->2, k=1->6, k=2->10
    #   i=3: k=0->3, k=1->7, k=2->11
    
    # M1 = (A11+A22)*B1: contributes to C_top (+1) and C_bot (+1)
    # M2 = (A12+A22)*B2: contributes to C_bot (-1)
    # M3 = A11*B2: contributes to C_top (+1)
    # M4 = A22*B1: contributes to C_top (-1) and C_bot (+1)
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    B1 = [0, 1, 2, 3, 4, 5]
    B2 = [6, 7, 8, 9, 10, 11]
    
    C_top = [0, 1, 4, 5, 8, 9]
    C_bot = [2, 3, 6, 7, 10, 11]
    
    # For each macro step, we use the 2x2*2x3 base decomposition
    # The base decomposition has U_base (4x11), V_base (6x11), W_base (6x11)
    # U_base[i*2+j, c] corresponds to A[i,j]
    # V_base[i*3+j, c] corresponds to B[i,j]
    # W_base[j*2+i, c] corresponds to C[i,j]
    
    macros = [
        # (A_blocks, A_signs, B_block, B_sign, [(C_block, sign), ...])
        ([A11, A22], [1, 1], B1, 1, [(C_top, 1), (C_bot, 1)]),   # M1
        ([A12, A22], [1, 1], B2, 1, [(C_bot, -1)]),               # M2
        ([A11], [1], B2, 1, [(C_top, 1)]),                        # M3
        ([A22], [1], B1, 1, [(C_top, -1), (C_bot, 1)]),           # M4
    ]
    
    col = 0
    for A_blocks, A_signs, B_block, B_sign, C_targets in macros:
        for c_inner in range(11):
            # U: for each A block, add A_sign * U_base
            for a_block, a_sign in zip(A_blocks, A_signs):
                for local_i in range(4):
                    U[a_block[local_i], col] += a_sign * U_base[local_i, c_inner]
            
            # V: B_block * B_sign * V_base
            for local_i in range(6):
                V[B_block[local_i], col] += B_sign * V_base[local_i, c_inner]
            
            # W: for each C target
            for C_block, C_sign in C_targets:
                for local_i in range(6):
                    W[C_block[local_i], col] += C_sign * W_base[local_i, c_inner]
            
            col += 1
    
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition correctness."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 100  # safety margin
    
    # Strategy: Split B into 4x4 + 4x3
    # 4x4 * 4x4: Strassen-based decomposition (rank 49)
    # 4x4 * 4x3: Strassen-based decomposition (rank 44)
    # Total rank: 49 + 44 = 93
    
    # Build 4x4 * 4x4 part
    U1, V1, W1 = _strassen_4x4_4x4()
    
    # Build 4x4 * 4x3 part
    U2, V2, W2 = _strassen_4x4_4x3()
    
    total_rank = U1.shape[1] + U2.shape[1]
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # First part: 4x4 * 4x4 (handles columns 0-3 of B)
    U[:, :U1.shape[1]] = U1
    V[:4*4, :U1.shape[1]] = V1
    W[:4*4, :U1.shape[1]] = W1
    
    # Second part: 4x4 * 4x3 (handles columns 4-6 of B)
    r1 = U1.shape[1]
    U[:, r1:] = U2
    # V2 is (12, 44), B rows 0-3, local cols 0-2 -> global cols 4-6
    for j in range(4):
        for d in range(3):
            V[j * 7 + 4 + d, r1:r1+U2.shape[1]] = V2[j * 3 + d, :]
    # W2 is (12, 44), local C cols 0-2 -> global C cols 4-6
    for d in range(3):
        for i in range(4):
            W[(4 + d) * 4 + i, r1:r1+U2.shape[1]] = W2[d * 4 + i, :]
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    if time.time() > deadline:
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
