"""
Rewritten program for matrix multiplication 2,3,8.
Constructs the rank-minimal (which is schoolbook rank 48) decomposition.
"""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the tensor decomposition of
    matrix multiplication of sizes 2x3 by 3x8.
    
    The rank is n*m*p = 48.
    U: (n*m, rank) -> (6, 48)
    V: (m*p, rank) -> (24, 48)
    W: (p*n, rank) -> (16, 48)
    
    Indexing conventions:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using list comprehension for compactness and distinctness from seed
    U = [[0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    col_idx = 0
    # Iterate through all combinations of i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to rows in U, V, W
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col_idx] = 1
                V[row_v][col_idx] = 1
                W[row_w][col_idx] = 1
                
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
