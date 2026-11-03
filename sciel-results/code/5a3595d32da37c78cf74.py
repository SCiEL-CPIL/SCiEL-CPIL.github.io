import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2 x 2x2 matrix multiplication using Strassen's algorithm.
    U: shape (4, 7), V: shape (4, 7), W: shape (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 terms for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3
    
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    W[0][0] = 1.0; W[3][0] = 1.0
    
    # M2: (A10+A11)B00 -> C10+1, C11-1
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    W[2][1] = 1.0; W[3][1] = -1.0
    
    # M3: A00(B01-B11) -> C01+1, C11+1
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    W[1][2] = 1.0; W[3][2] = 1.0
    
    # M4: A11(B10-B00) -> C00+1, C10+1
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    W[0][3] = 1.0; W[2][3] = 1.0
    
    # M5: (A00+A01)B11 -> C00-1, C01+1
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0; W[1][4] = 1.0
    
    # M6: (A10-A00)(B00+B01) -> C11+1
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    W[3][5] = 1.0
    
    # M7: (A01-A11)(B10+B11) -> C00+1
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    W[0][6] = 1.0
    
    return U, V, W

def _strassen_4x4x4():
    """
    Returns U, V, W for 4x4 x 4x4 matrix multiplication using recursive Strassen.
    U: shape (16, 49), V: shape (16, 49), W: shape (16, 49)
    """
    U2, V2, W2 = _strassen_2x2x2()
    rank_2 = 7
    rank_4 = 7 * rank_2  # 49
    
    U = [[0.0]*rank_4 for _ in range(16)]
    V = [[0.0]*rank_4 for _ in range(16)]
    W = [[0.0]*rank_4 for _ in range(16)]
    
    # For 4x4, we have 4 sub-blocks of 2x2
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # Similarly for B and C
    
    # Define the 7 Strassen terms for 4x4 (same structure as 2x2 but with 2x2 blocks)
    # Each term uses the 2x2x2 Strassen decomposition
    
    # Block index to global index mapping for 2x2 block at position (bi, bj) in 4x4
    def block_to_global(bi, bj, local_idx):
        # local_idx: 0=A00, 1=A01, 2=A10, 3=A11 within the 2x2 block
        lr = local_idx // 2
        lc = local_idx % 2
        global_r = bi * 2 + lr
        global_c = bj * 2 + lc
        return global_r * 4 + global_c
    
    # The 7 terms for 4x4 Strassen (same as 2x2 but blocks are 2x2)
    # M1: (A00+A11)(B00+B11) -> C00+1, C11+1
    # M2: (A10+A11)B00 -> C10+1, C11-1
    # M3: A00(B01-B11) -> C01+1, C11+1
    # M4: A11(B10-B00) -> C00+1, C10+1
    # M5: (A00+A01)B11 -> C00-1, C01+1
    # M6: (A10-A00)(B00+B01) -> C11+1
    # M7: (A01-A11)(B10+B11) -> C00+1
    
    terms = [
        # (A_blocks, B_blocks, C_blocks) where each block is (bi, bj, coeff)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),      # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),               # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),               # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),               # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),               # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),               # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])                # M7
    ]
    
    for t_idx, (a_blocks, b_blocks, c_blocks) in enumerate(terms):
        for r in range(rank_2):
            col = t_idx * rank_2 + r
            
            # Process U
            for (bi, bj, coeff_a) in a_blocks:
                for local_idx in range(4):
                    gi = block_to_global(bi, bj, local_idx)
                    U[gi][col] += coeff_a * U2[local_idx][r]
            
            # Process V
            for (bi, bj, coeff_b) in b_blocks:
                for local_idx in range(4):
                    gi = block_to_global(bi, bj, local_idx)
                    V[gi][col] += coeff_b * V2[local_idx][r]
            
            # Process W
            for (bi, bj, coeff_c) in c_blocks:
                for local_idx in range(4):
                    gi = block_to_global(bi, bj, local_idx)
                    W[gi][col] += coeff_c * W2[local_idx][r]
    
    return U, V, W

def _strassen_4x4x1():
    """
    Returns U, V, W for 4x4 x 4x1 matrix multiplication.
    This is a matrix-vector product: C = A * b where A is 4x4, b is 4x1, C is 4x1.
    We can use a rank-7 decomposition based on Strassen.
    U: shape (16, 7), V: shape (4, 7), W: shape (4, 7)
    """
    # For matrix-vector product 4x4 * 4x1:
    # We can split A into 2x2 blocks and b into 2x1 blocks
    # C = [A00 b0; A10 b0] + [A01 b1; A11 b1]
    # 
    # Using Strassen-like tricks, we can compute this with 7 multiplications
    # Each "multiplication" is a 2x2 * 2x1 product which has rank 2 (schoolbook)
    # But we can do better with Winograd-style approaches
    #
    # Actually, for 2x2 * 2x1, the rank is 2 (it's just two dot products)
    # So total rank would be 7 * 2 = 14? That's worse than schoolbook (16).
    #
    # Let me use a different approach: direct rank-7 for 4x4x1
    # This is equivalent to computing 4 linear combinations of 4 variables each
    # with a special structure.
    
    # For now, use schoolbook for 4x4x1 (rank 16)
    # But we can try to find a better decomposition
    # Actually, the rank of 4x4x1 tensor is 16 (it's just the standard matrix-vector product)
    # No improvement possible for square matrix-vector product in general
    
    # Let's just use schoolbook
    rank = 16
    U = [[0.0]*rank for _ in range(16)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    col = 0
    for i in range(4):
        for k in range(4):
            U[i*4+k][col] = 1.0
            V[k][col] = 1.0
            W[i][col] = 1.0
            col += 1
    
    return U, V, W

def _strassen_4x4x5():
    """
    Returns U, V, W for 4x4 x 4x5 matrix multiplication.
    We decompose this as 5 separate 4x4x1 multiplications.
    U: shape (16, 80), V: shape (20, 80), W: shape (20, 80)
    """
    U1, V1, W1 = _strassen_4x4x1()
    rank_1 = len(U1[0])  # 16
    
    total_rank = 5 * rank_1  # 80
    U = [[0.0]*total_rank for _ in range(16)]
    V = [[0.0]*total_rank for _ in range(20)]
    W = [[0.0]*total_rank for _ in range(20)]
    
    for j in range(5):
        for c in range(rank_1):
            col = j * rank_1 + c
            for i in range(16):
                U[i][col] = U1[i][c]
            for k in range(4):
                V[k*5+j][col] = V1[k][c]
            for i in range(4):
                W[j*4+i][col] = W1[i][c]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Get rank-49 decomposition for 4x4x4
    U4, V4, W4 = _strassen_4x4x4()
    rank_4 = len(U4[0])  # Should be 49
    
    # Get decomposition for 4x4x5
    U5, V5, W5 = _strassen_4x4x5()
    rank_5 = len(U5[0])  # Should be 80
    
    # Combine for 4x4x9
    total_rank = rank_4 + rank_5
    U = [[0.0] * total_rank for _ in range(n * m)]  # 16 x total_rank
    V = [[0.0] * total_rank for _ in range(m * p)]  # 36 x total_rank
    W = [[0.0] * total_rank for _ in range(p * n)]  # 36 x total_rank
    
    # Copy 4x4x4 part (first 4 columns of B and C)
    for i in range(16):
        for c in range(rank_4):
            U[i][c] = U4[i][c]
            
    # V4 corresponds to B1 (4x4), occupies first 4 columns of B
    for k in range(4):
        for j in range(4):
            v_idx_4 = k * 4 + j
            v_idx_9 = k * 9 + j
            for c in range(rank_4):
                V[v_idx_9][c] = V4[v_idx_4][c]
                
    # W4 corresponds to C1 (4x4), occupies first 4 columns of C
    for j in range(4):
        for i in range(4):
            w_idx_4 = j * 4 + i
            w_idx_9 = j * 9 + i
            for c in range(rank_4):
                W[w_idx_9][c] = W4[w_idx_4][c]
                
    # Copy 4x4x5 part (last 5 columns of B and C)
    for i in range(16):
        for c in range(rank_5):
            U[i][rank_4 + c] = U5[i][c]
            
    # V5 corresponds to B2 (4x5), occupies last 5 columns of B
    for k in range(4):
        for j_local in range(5):
            v_idx_5 = k * 5 + j_local
            v_idx_9 = k * 9 + (j_local + 4)
            for c in range(rank_5):
                V[v_idx_9][rank_4 + c] = V5[v_idx_5][c]
                
    # W5 corresponds to C2 (4x5), occupies last 5 columns of C
    for j_local in range(5):
        for i in range(4):
            w_idx_5 = j_local * 4 + i
            w_idx_9 = (j_local + 4) * 4 + i
            for c in range(rank_5):
                W[w_idx_9][rank_4 + c] = W5[w_idx_5][c]
                
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_s[i * m + k][col] = 1
                    V_s[k * p + j][col] = 1
                    W_s[j * n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
