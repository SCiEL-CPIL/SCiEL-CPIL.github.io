import numpy as np
import time
import random

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    target_rank = 10
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds limit
    
    # Define the target tensor T
    # T[alpha, beta, gamma] = 1 if alpha=(i,j), beta=(j',k), gamma=(k',i') and i=i', j=j', k=k'
    # We can represent T as a list of tuples (i, j, k) where the entry is 1.
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # alpha = i*m + j
                # beta = j*p + k
                # gamma = k*n + i
                target_entries.append((i*m + j, j*p + k, k*n + i))
                
    def check_decomposition(U, V, W, rank):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Check if sum_c U[alpha, c] V[beta, c] W[gamma, c] == T[alpha, beta, gamma]
        # For each target entry (alpha, beta, gamma), sum should be 1.
        # For all other entries, sum should be 0.
        
        # Efficient check:
        # Compute the tensor product sum for all non-zero targets.
        # Also check a few random non-target entries to ensure they are 0.
        
        # 1. Check target entries
        for alpha, beta, gamma in target_entries:
            val = 0
            for c in range(rank):
                val += U[alpha, c] * V[beta, c] * W[gamma, c]
            if not np.isclose(val, 1.0):
                return False
                
        # 2. Check non-target entries (sampling)
        # Total entries: (n*m) * (m*p) * (p*n) = 4 * 6 * 6 = 144
        # Target entries: 12
        # Non-target: 132
        # We can check all of them if it's fast enough.
        for alpha in range(n*m):
            for beta in range(m*p):
                for gamma in range(p*n):
                    if (alpha, beta, gamma) in target_entries:
                        continue
                    val = 0
                    for c in range(rank):
                        val += U[alpha, c] * V[beta, c] * W[gamma, c]
                    if not np.isclose(val, 0.0):
                        return False
        return True

    # Search for rank 10
    # We will try random integer matrices
    # To speed up, we can fix W to be identity-like or sparse?
    # Let's just try random small integers.
    
    found = None
    
    # Try rank 10
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        # Generate random U, V, W with entries in {-1, 0, 1}
        # Scale might be needed?
        U = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
        V = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
        W = np.array([[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)
        
        if check_decomposition(U, V, W, target_rank):
            found = (U, V, W, target_rank)
            break
            
    if found:
        U, V, W, rank = found
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # If rank 10 not found, try rank 11?
    # Or just return seed (rank 12)
    
    # Fallback to seed
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
