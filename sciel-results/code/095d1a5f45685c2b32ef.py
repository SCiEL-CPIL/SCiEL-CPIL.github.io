import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor T is defined by T[i, j, k] = 1 for all i, j, k
    # The decomposition condition is sum_r U[i, j, r] V[j, k, r] W[i, k, r] = 1
    # Note: The code uses W[k*n+i], which is index (k, i).
    # The standard matrix multiplication tensor has W[i, k] or W[k, i] depending on convention.
    # The check in the code is:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    
    # We try to find a decomposition of rank r
    # Lower bound is max(n*m, m*p, p*n) = 12
    # Schoolbook is 36.
    
    max_rank = n * m * p
    found_solution = None
    
    def verify(U, V, W, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r_idx in range(r):
                        val += U[i * m + j, r_idx] * V[j * p + k, r_idx] * W[k * n + i, r_idx]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def is_half_integer(arr, tol=1e-6):
        doubled = 2 * arr
        rounded = np.round(doubled)
        if np.allclose(doubled, rounded, atol=tol):
            return rounded / 2.0
        return None

    # Try different ranks
    # We start from a lower bound.
    # For 3x3x4, rank is likely > 20.
    for r in range(20, max_rank + 1):
        # Try a few random initializations with small integers
        for attempt in range(50):
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Initialize with random small integers
            for idx in range(n * m):
                for r_idx in range(r):
                    if random.random() < 0.5:
                        U[idx, r_idx] = random.choice([-1, 1])
            for idx in range(m * p):
                for r_idx in range(r):
                    if random.random() < 0.5:
                        V[idx, r_idx] = random.choice([-1, 1])
            for idx in range(p * n):
                for r_idx in range(r):
                    if random.random() < 0.5:
                        W[idx, r_idx] = random.choice([-1, 1])
                        
            # Check if this random initialization works (unlikely)
            if verify(U, V, W, r):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook algorithm
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
    return {"U": U, "V": V, "W": W}
