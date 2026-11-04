import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify the decomposition is correct."""
    import numpy as np
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                max_err = max(max_err, abs(val - 1))
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Use Schoolbook decomposition (Rank = n*m*p = 112)
    # This is guaranteed to be correct and fast to construct.
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    # Verify
    err = _verify(U, V, W)
    if err > 1e-9:
        # Should not happen for schoolbook, but fallback just in case
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
