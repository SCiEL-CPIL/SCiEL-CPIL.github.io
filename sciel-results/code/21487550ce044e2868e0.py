"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # We use a block-partitioning strategy:
    # Split j (size 4) into j0 (size 1) and j1 (size 3)
    # Split k (size 5) into k0 (size 1) and k1 (size 4)
    # This gives 4 blocks:
    # Block 1: i in [0,1], j=0, k=0 -> 2x1x1, rank 2
    # Block 2: i in [0,1], j in [1,2,3], k=0 -> 2x3x1, rank 6
    # Block 3: i in [0,1], j=0, k in [1,2,3,4] -> 2x1x4, rank 8
    # Block 4: i in [0,1], j in [1,2,3], k in [1,2,3,4] -> 2x3x4, rank 4
    # Total rank = 2 + 6 + 8 + 4 = 20
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]  # 8 x 20
    V = [[0] * rank for _ in range(m * p)]  # 20 x 20
    W = [[0] * rank for _ in range(p * n)]  # 10 x 20
    
    col = 0
    
    # Block 1: i in [0,1], j=0, k=0 -> 2x1x1, rank 2
    # C[i,0] = A[i,0] * B[0,0]
    for i in range(n):
        U[i * m + 0][col] = 1
        V[0 * p + 0][col] = 1
        W[0 * n + i][col] = 1
        col += 1
    
    # Block 2: i in [0,1], j in [1,2,3], k=0 -> 2x3x1, rank 6
    # C[i,0] += sum_{j=1}^{3} A[i,j] * B[j,0]
    # This is just schoolbook for this sub-block
    for i in range(n):
        for j_idx in range(3):
            j = 1 + j_idx
            U[i * m + j][col] = 1
            V[j * p + 0][col] = 1
            W[0 * n + i][col] = 1
            col += 1
    
    # Block 3: i in [0,1], j=0, k in [1,2,3,4] -> 2x1x4, rank 8
    # C[i,k] = A[i,0] * B[0,k] for k in [1,2,3,4]
    # This is just schoolbook for this sub-block
    for i in range(n):
        for k_idx in range(4):
            k = 1 + k_idx
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 4: i in [0,1], j in [1,2,3], k in [1,2,3,4] -> 2x3x4
    # C[i,k] = sum_{j=1}^{3} A[i,j] * B[j,k] for i in [0,1], k in [1,2,3,4]
    # We need a rank-4 decomposition for this 2x3x4 sub-problem
    # Let A_sub be 2x3, B_sub be 3x4, C_sub be 2x4
    # C_sub = A_sub * B_sub
    # We can use the fact that this is equivalent to computing 8 values from 12 inputs
    # with 3 shared intermediate variables (the j indices)
    
    # One approach: use 4 factors
    # Factor 1: handles A[0,1]*B[1,1], A[0,2]*B[2,1], A[0,3]*B[3,1], A[1,1]*B[1,2], A[1,2]*B[2,2], A[1,3]*B[3,2]
    # This doesn't work cleanly. Let me think differently.
    
    # Actually, for a 2x3x4 multiplication, the minimal rank is at least max(2*4, 3*4, 2*3) = 12? No.
    # The rank of the tensor M_{2,3,4} is at least the max of the dimensions of the output, which is 8.
    # But we need rank 4, which is less than 8, so it's impossible with exact decomposition.
    
    # Let me reconsider. The total rank should be 20, but if block 4 requires rank >= 8, then total is >= 24.
    # So rank 20 is not achievable with this partitioning.
    
    # Let me use a different approach. Use schoolbook for everything but optimize.
    # Actually, let me just use the schoolbook rank 40 to be safe.
    
    # Reset and use schoolbook
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
