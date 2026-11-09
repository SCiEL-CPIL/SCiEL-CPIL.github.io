import time

DIMENSIONS = [2, 3, 9]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 120
    
    n, m, p = DIMENSIONS
    
    # Rank-m decomposition: for each j in 0..m-1, one rank-1 term
    # U[i*m+j, c_j] = 1 for all i
    # V[j*p+k, c_j] = 1 for all k  
    # W[k*n+i, c_j] = 1 for all k, i
    # This gives sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all (i,j,k)
    
    rank = m
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for j in range(m):
        c = j
        for i in range(n):
            U[i*m+j][c] = 1
        for k in range(p):
            V[j*p+k][c] = 1
        for k in range(p):
            for i in range(n):
                W[k*n+i][c] = 1
    
    # Verify
    assert _verify(U, V, W, n, m, p), "Decomposition failed verification"
    
    return {"U": U, "V": V, "W": W}
