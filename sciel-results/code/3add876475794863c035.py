import numpy as np
import random
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    R_target = 29
    R_fallback = 30
    
    # Time limit for search
    start_time = time.time()
    time_limit = 5.0 # 5 seconds
    
    # Try to find a rank 29 solution
    # We use a randomized search with small integer/half-integer values
    # This is not exhaustive, but it's the only feasible way to potentially beat 1.0
    # within the time limit.
    
    # The space is too large for exhaustive search.
    # We will try a few random initializations and local improvements.
    
    # However, to respect the "deterministic" part of the directive as much as possible
    # without timing out, we can fix a deterministic seed for the random generator.
    
    random.seed(42)
    
    found = False
    while time.time() - start_time < time_limit and not found:
        R = R_target
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Initialize with random small integers
        # This is a heuristic. If it doesn't work, we fallback.
        for r in range(R):
            for i in range(n*m):
                U[i, r] = random.choice([-1, 0, 1, 0.5, -0.5])
            for j in range(m*p):
                V[j, r] = random.choice([-1, 0, 1, 0.5, -0.5])
            for k in range(p*n):
                W[k, r] = random.choice([-1, 0, 1, 0.5, -0.5])
        
        # Verify
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Target T[i, j, k] = 1
        
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            found = True
            break

    if found:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback Rank 30
        R = R_fallback
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
