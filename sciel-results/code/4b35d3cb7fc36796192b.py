"""Find rank-minimal exact tensor decomposition for 2x5 by 5x5 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [2, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Tensor: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for i2 in range(n):
                            for k2 in range(p):
                                # Compute sum_c U[i*m+j, c] * V[j2*p+k, c] * W[k2*n+i2, c]
                                val = 0
                                for c in range(r):
                                    val += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i2][c]
                                expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                                if abs(val - expected) > 1e-9:
                                    return False
        return True
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W
    
    U_sb, V_sb, W_sb = schoolbook()
    best = (len(U_sb[0]), U_sb, V_sb, W_sb)
    
    # Try to find lower rank by random search
    # Strategy: try random rank-r decompositions and check validity
    # This is expensive, so limit attempts
    
    for r in range(25, 50):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        # Try a few random attempts
        for attempt in range(20):
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            # Random initialization
            U = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(r)] for _ in range(p*n)]
            if check_decomposition(U, V, W):
                best = (r, U, V, W)
                break
        if best[0] < 50:
            break
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
