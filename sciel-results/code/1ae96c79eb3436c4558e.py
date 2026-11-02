"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify_numpy(U, V, W, n, m, p, tol=1e-9):
        """Verify tensor decomposition using numpy for speed."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        
        # Build full tensor T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for c in range(r):
            T += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        
        # Check diagonal entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                        return False
        
        # Check off-diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    for j2 in range(m):
                        for k2 in range(p):
                            for i2 in range(n):
                                if i == i2 and j == j2 and k == k2:
                                    continue
                                b = j2*p + k2
                                c_idx = k2*n + i2
                                if abs(T[a, b, c_idx]) > tol:
                                    return False
        
        return True
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Use schoolbook rank 50
    U, V, W = schoolbook(n, m, p)
    
    # Verify before returning
    if verify_numpy(U, V, W, n, m, p, tol=1e-9):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
