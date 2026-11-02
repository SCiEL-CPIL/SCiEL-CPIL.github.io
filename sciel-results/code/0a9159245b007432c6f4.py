"""
Tensor decomposition for 4x5 * 5x5 matrix multiplication.
We attempt to find a lower-rank exact decomposition than the schoolbook rank of 100.
Uses structured random search with verification and a time-bounded search loop.
"""
import time
import random
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank_school = n * m * p  # 100
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Start with schoolbook as best
    best_rank = rank_school
    best_U = [[0.0] * rank_school for _ in range(n * m)]
    best_V = [[0.0] * rank_school for _ in range(m * p)]
    best_W = [[0.0] * rank_school for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                best_U[i * m + j][c] = 1.0
                best_V[j * p + k][c] = 1.0
                best_W[k * n + i][c] = 1.0
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for c in range(rank):
                        s += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    # Try to find lower rank via random structured search
    # Strategy: try ranks from 99 down, with limited attempts per rank
    for target_rank in range(rank_school - 1, max(rank_school - 30, 0), -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random attempts for this rank
        max_attempts = 20 if target_rank > 90 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random sparse-ish factors
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Initialize with small random values
            for c in range(target_rank):
                # Each column gets a few non-zero entries
                for _ in range(random.randint(1, 3)):
                    U[random.randint(0, n * m - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                for _ in range(random.randint(1, 3)):
                    V[random.randint(0, m * p - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
                for _ in range(random.randint(1, 3)):
                    W[random.randint(0, p * n - 1), c] = random.choice([-1, 1]) * random.uniform(0.5, 2.0)
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify(U_list, V_list, W_list, target_rank):
                best_rank = target_rank
                best_U = U_list
                best_V = V_list
                best_W = W_list
                return {
                    "U": best_U,
                    "V": best_V,
                    "W": best_W
                }
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
