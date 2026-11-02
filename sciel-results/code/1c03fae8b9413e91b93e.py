"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
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
        return {"U": U, "V": V, "W": W}
    
    def is_exact(U, V, W, n, m, p):
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(r):
                        s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
        
    def try_random_rank(r, num_trials=200):
        """Try to find a rank-r decomposition using random search."""
        for trial in range(num_trials):
            # Randomly initialize U, V, W with small integers
            # Using {-1, 0, 1} is standard for exact integer decompositions
            U = np.random.choice([-1, 0, 1], size=(n * m, r))
            V = np.random.choice([-1, 0, 1], size=(m * p, r))
            W = np.random.choice([-1, 0, 1], size=(p * n, r))
            
            # Check if this decomposition is exact
            if is_exact(U, V, W, n, m, p):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None

    # Try to find a rank lower than schoolbook (60)
    # We start from 59 down to 50 to save time, as lower ranks are harder to find randomly
    # If we find one, we return it immediately.
    for r in range(59, 49, -1):
        res = try_random_rank(r, num_trials=100)
        if res:
            return res
            
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
