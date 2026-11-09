"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS
    
    # Naive decomposition: rank = m * p = 4 * 8 = 32
    # For each (j, k) pair, create a rank-1 term
    rank = m * p  # 32
    
    U = [[0] * rank for _ in range(n * m)]  # (n*m) x rank = 8 x 32
    V = [[0] * rank for _ in range(m * p)]  # (m*p) x rank = 32 x 32
    W = [[0] * rank for _ in range(p * n)]  # (p*n) x rank = 16 x 32
    
    col = 0
    for j in range(m):
        for k in range(p):
            # U: A[i, j] coefficient is 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # V: B[j, k] coefficient is 1
            V[j * p + k][col] = 1
            # W: C[i, k] coefficient is 1 for all i
            # W is indexed as W[k*n + i, r]
            for i in range(n):
                W[k * n + i][col] = 1
            col += 1
    
    if time.time() > deadline:
        # Should not happen for this simple construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
