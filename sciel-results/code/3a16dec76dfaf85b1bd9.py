"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal: for all (i,j,k) != (i3,j2,k3), sum should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i3 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i3 and j == j2 and k == k2:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k2][c] * W[k*n+i3][c]
                                if abs(s) > tol:
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
    
    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Vectorized verification using numpy."""
        r = len(U[0])
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                    if abs(s - 1.0) > tol:
                        return False
        # Check off-diagonal
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i3 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if i == i3 and j == j2 and k == k2:
                                    continue
                                s = np.sum(U_np[i*m+j] * V_np[j2*p+k2] * W_np[k*n+i3])
                                if abs(s) > tol:
                                    return False
        return True
    
    # Rank-25 decomposition for 2x5 @ 5x5
    # Key insight: The 25 rank-1 terms correspond to the 25 (j,k) pairs.
    # For each (j0, k0):
    #   U[i*5+j, j0*5+k0] = delta(j, j0)   -- selects column j0 from A
    #   V[j*5+k, j0*5+k0] = delta(j, j0) * delta(k, k0)  -- selects B[j0, k0]
    #   W[k*2+i, j0*5+k0] = delta(k, k0)   -- selects output column k0
    #
    # This works because both rows of A multiply the same B, so the decomposition
    # is shared across i. The i-index in W[k*n+i] is handled by W depending only
    # on k (not i), and U depending only on j (not i).
    
    rank = m * p  # 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j0 in range(m):
        for k0 in range(p):
            c = j0 * p + k0
            # U: for each row i, U[i*m+j, c] = 1 if j == j0
            for i in range(n):
                U[i * m + j0][c] = 1
            # V: V[j*p+k, c] = 1 if j == j0 and k == k0
            V[j0 * p + k0][c] = 1
            # W: W[k*n+i, c] = 1 if k == k0 (for all i)
            for i in range(n):
                W[k0 * n + i][c] = 1
    
    if verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 50
    U, V, W = schoolbook(n, m, p)
    if verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Last resort: return schoolbook even if verification failed
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
