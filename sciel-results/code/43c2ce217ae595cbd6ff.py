"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block-wise construction approach
    # We decompose the multiplication into m blocks of size n x 1 and 1 x p
    # Each block j corresponds to the j-th column of A and j-th row of B
    block_size = n * p
    for j in range(m):
        # Offset for this block in the rank dimension
        offset = j * block_size
        
        # Construct U, V, W for this block
        # U has 1s at (i, j) for all i
        # V has 1s at (j, k) for all k
        # W has 1s at (k, i) for all i, k
        
        for i in range(n):
            u_idx = i * m + j
            for k in range(p):
                local_idx = i * p + k
                global_idx = offset + local_idx
                
                U[u_idx][global_idx] = 1
                V[j * p + k][global_idx] = 1
                W[k * n + i][global_idx] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
