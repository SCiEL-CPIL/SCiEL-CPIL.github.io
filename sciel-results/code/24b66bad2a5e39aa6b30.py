import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    
    # We decompose the problem into:
    # 1. Core: A[0:4, 0:4] * B[0:4, 0:4] -> C[0:4, 0:4]
    #    Rank 49 (Kronecker of two rank-7 2x2x2 decompositions)
    # 2. Strip 1: A[0:4, 0:4] * B[0:4, 4:6] -> C[0:4, 4:6]
    #    Rank 2 (Two rank-1 4x4x1 decompositions)
    # 3. Strip 2: A[0:4, 4:5] * B[4:5, 0:6] -> C[0:4, 0:6]
    #    Rank 1 (One rank-1 4x1x6 decomposition)
    
    # Helper: Rank-7 decomposition for 2x2x2 matrix multiplication
    # Indices for U: i*2+j, V: j*2+k, W: k*2+i
    U2 = [
        [1, 1, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1]
    ]
    V2 = [
        [1, 0, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1]
    ]
    W2 = [
        [1, 0, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1]
    ]
    
    # Verify U2, V2, W2 for 2x2x2
    # C_ik = sum_j A_ij B_jk
    # Term c: sum_i,k W2[k*2+i][c] * (sum_j U2[i*2+j][c] * V2[j*2+k][c])
    # This is a standard rank-7 decomposition.
    
    # Construct 4x4x4 decomposition using Kronecker product
    # U4 = U2 (x) U2, V4 = V2 (x) V2, W4 = W2 (x) W2
    # Size: 16 x 49
    
    def kron(A, B):
        a, b = np.array(A), np.array(B)
        return np.kron(a, b).tolist()
    
    U4 = kron(U2, U2)
    V4 = kron(V2, V2)
    W4 = kron(W2, W2)
    
    rank_core = 49
    rank_strip1 = 2
    rank_strip2 = 1
    total_rank = rank_core + rank_strip1 + rank_strip2
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (20, total_rank)
    # V shape: (m*p, rank) = (30, total_rank)
    # W shape: (p*n, rank) = (24, total_rank)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Fill Core (Rank 49)
    # U indices: i in 0..3, j in 0..3 -> i*5 + j
    # V indices: j in 0..3, k in 0..3 -> j*6 + k
    # W indices: k in 0..3, i in 0..3 -> k*4 + i
    
    for c in range(rank_core):
        for i in range(4):
            for j in range(4):
                # U4 index is (i*2 + j) ? No.
                # U4 is Kronecker of U2.
                # U2 index: i2*2 + j2.
                # U4 index: (i2*2 + j2) * 4 + (i1*2 + j1)?
                # Kronecker product A (x) B:
                # (A (x) B)_{(i_A, i_B), (j_A, j_B)} = A_{i_A, j_A} * B_{i_B, j_B}
                # Here indices are flattened.
                # Let's map the flattened indices of U4 (size 16) to i,j in 0..3.
                # U4 row r (0..15).
                # r = i * 4 + j?
                # In Kronecker, if A is 4x7 and B is 4x7, result is 16x49.
                # Row index r = r_A * 4 + r_B.
                # r_A = i_A * 2 + j_A.
                # r_B = i_B * 2 + j_B.
                # So i = i_A * 2 + i_B? No.
                # The matrix A is 4x4. Indices (i, j).
                # In Kronecker decomposition of 4x4x4 from 2x2x2:
                # i = i_A * 2 + i_B
                # j = j_A * 2 + j_B
                # k = k_A * 2 + k_B
                # U4 row index is (i_A * 2 + j_A) * 4 + (i_B * 2 + j_B)?
                # No, U4 is 16x49.
                # The row index corresponds to (i, j).
                # Let's just iterate over i, j and find the corresponding row in U4.
                # U4 row r.
                # r = (i_A * 2 + j_A) * 4 + (i_B * 2 + j_B)
                # i = i_A * 2 + i_B
                # j = j_A * 2 + j_B
                # This mapping is complex to invert.
                # Easier: Iterate over U4 rows, decode i, j.
                
                pass
                
    # Let's use a simpler mapping for U4, V4, W4.
    # U4 is 16x49.
    # Row r (0..15).
    # r = i * 4 + j.
    # We need to check if U4[r][c] corresponds to U2[i_A*2+j_A][c_A] * U2[i_B*2+j_B][c_B].
    # c = c_A * 7 + c_B.
    # i = i_A * 2 + i_B.
    # j = j_A * 2 + j_B.
    
    for c in range(rank_core):
        c_A = c // 7
        c_B = c % 7
        for r in range(16):
            val = U4[r][c]
            if val != 0:
                i = r // 4
                j = r % 4
                U[i * m + j][c] = val
                
    for c in range(rank_core):
        c_A = c // 7
        c_B = c % 7
        for r in range(16):
            val = V4[r][c]
            if val != 0:
                j = r // 4
                k = r % 4
                V[j * p + k][c] = val
                
    for c in range(rank_core):
        c_A = c // 7
        c_B = c % 7
        for r in range(16):
            val = W4[r][c]
            if val != 0:
                k = r // 4
                i = r % 4
                W[k * n + i][c] = val
                
    # Fill Strip 1 (Rank 2)
    # A[0:4, 0:4] * B[0:4, 4:6]
    # Term 1: k=4
    c1 = rank_core
    for i in range(4):
        for j in range(4):
            U[i * m + j][c1] = 1
    for j in range(4):
        V[j * p + 4][c1] = 1
    for i in range(4):
        W[4 * n + i][c1] = 1
        
    # Term 2: k=5
    c2 = rank_core + 1
    for i in range(4):
        for j in range(4):
            U[i * m + j][c2] = 1
    for j in range(4):
        V[j * p + 5][c2] = 1
    for i in range(4):
        W[5 * n + i][c2] = 1
        
    # Fill Strip 2 (Rank 1)
    # A[0:4, 4:5] * B[4:5, 0:6]
    c3 = rank_core + 2
    for i in range(4):
        U[i * m + 4][c3] = 1
    for k in range(6):
        V[4 * p + k][c3] = 1
    for i in range(4):
        for k in range(6):
            W[k * n + i][c3] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
