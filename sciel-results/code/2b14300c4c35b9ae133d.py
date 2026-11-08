"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Naive (Standard) Matrix Multiplication Decomposition
    # Rank = n * m * p
    # This approach uses the trivial tensor decomposition where each
    # multiplication step corresponds to a single term in the sum.
    # It is a distinct approach family from block-based algebraic tricks
    # like Strassen's algorithm.
    
    total_rank = n * m * p
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U maps A[i,j] to the idx-th product
                U[i * m + j][idx] = 1
                # V maps B[j,k] to the idx-th product
                V[j * p + k][idx] = 1
                # W maps the idx-th product to C[k,i] (trace convention W[k*n+i, r])
                W[k * n + i][idx] = 1
                idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
