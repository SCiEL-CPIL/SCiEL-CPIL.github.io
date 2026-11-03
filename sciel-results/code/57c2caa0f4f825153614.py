import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-41 decomposition for matrix multiplication of size 2x3x10.
    
    Strategy:
    Split A (2x3) into A_L (2x2) and A_R (2x1).
    Split B (3x10) into B_L (2x10) and B_R (1x10).
    C = A_L B_L + A_R B_R.
    
    A_R B_R is a rank-1 tensor (outer product of column 2 of A and row 2 of B).
    A_L B_L consists of 10 independent 2x2x1 multiplications (one for each column of B_L).
    Each 2x2x1 multiplication has schoolbook rank 4.
    Total rank = 1 + 10 * 4 = 41.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    
    # Rank calculation
    # A_R B_R: rank 1
    # A_L B_L: 10 columns, each is M_2,2,1 (rank 4)
    rank = 1 + p * 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_R B_R (Rank 1)
    # A_R is column j=2 of A. B_R is row l=2 of B.
    # Contribution to C[i,k] is A[i,2] * B[2,k]
    
    # U[i*m + 2, col] = 1 for i in 0..1
    for i in range(n):
        U[i * m + 2][col] = 1
        
    # V[2*p + k, col] = 1 for k in 0..9
    for k in range(p):
        V[2 * p + k][col] = 1
        
    # W[k*n + i, col] = 1 for all i, k
    for i in range(n):
        for k in range(p):
            W[k * n + i][col] = 1
            
    col += 1
    
    if time.time() > deadline:
        return {"U": U, "V": W} # Error handling, shouldn't happen

    # Part 2: A_L B_L (Rank 40)
    # A_L is columns j=0,1 of A. B_L is rows l=0,1 of B.
    # For each k in 0..9, we compute C[:,k] = A_L[:,0] * B_L[0,k] + A_L[:,1] * B_L[1,k]
    # This is M_2,2,1 for the k-th column.
    # Schoolbook for M_2,2,1 (indices i, j, k_local=0):
    # 1. A[0,0] * B[0,0] -> C[0,0]
    # 2. A[0,1] * B[1,0] -> C[0,0]
    # 3. A[1,0] * B[0,0] -> C[1,0]
    # 4. A[1,1] * B[1,0] -> C[1,0]
    
    for k in range(p):
        # Term 1: A[0,0] * B[0,k] -> C[0,k]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> C[0,k]
        U[0 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> C[1,k]
        U[1 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Truncate if necessary, though loop is fast
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
