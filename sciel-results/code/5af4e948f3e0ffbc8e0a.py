"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # safety margin
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    def verify(U, V, W):
        """Check if the decomposition is valid."""
        r = len(U[0])
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    best = schoolbook()
    
    # Try to improve via random search
    # Strategy: start from schoolbook, try to merge columns
    # Two columns c1, c2 can be merged if there exist scalars such that
    # the resulting decomposition is still valid.
    
    # Actually, a simpler approach: try random perturbations and projection
    
    # Let me try a different approach: use the fact that we can express
    # the tensor as a sum of rank-1 terms, and try to find a smaller set.
    
    # For now, return schoolbook. In a future iteration, we could try
    # more sophisticated methods.
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
