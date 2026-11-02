"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    """
    Tensor decomposition for 2x2 by 2x12 matrix multiplication.
    Uses a 'row-aggregation' style decomposition where we treat the two output rows
    independently. Since the input is 2x2, each output row depends on both input rows.
    
    For each output row i (0, 1), we need to compute sum_j A[i,j] * B[j,:].
    This is a sum of 2 vector-matrix multiplications (1x2 by 2x12).
    Each 1x2 by 2x12 multiplication has rank 2*12 = 24 (schoolbook).
    Total rank = 2 (output rows) * 24 = 48.
    
    However, let's double check if we can do better.
    The standard rank for m x n by n x p is m*n*p.
    Here m=2, n=2, p=12. Rank = 2*2*12 = 48.
    
    The previous attempt at rank 24 was rejected or invalid.
    The schoolbook rank 48 is valid.
    Let's try to be slightly more efficient in construction, but the rank remains 48.
    """
    n, m, p = DIMENSIONS
    # Note: In the task description, U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # Wait, the task says:
    # U: (n*m, rank) -> rows indexed by (i, j) where i in [0,n), j in [0,m)?
    # Let's check the seed code:
    # U[i*m+j][column] = 1
    # V[j*p+k][column] = 1
    # W[k*n+i][column] = 1
    #
    # This corresponds to:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    #
    # The tensor T_{i,j,k} = A[i,j] B[j,k] contributes to C[i,k].
    #
    # U has dimension n*m. Index is i*m + j.
    # V has dimension m*p. Index is j*p + k.
    # W has dimension p*n. Index is k*n + i.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
