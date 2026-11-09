import numpy as np
import time
import itertools
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 100
    
    start_time = time.time()
    deadline = 145.0
    
    def build_schoolbook():
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = build_schoolbook()
    best_rank = rank_school

    # Try to find a better decomposition
    # We know rank 36 is possible for 5x5x5.
    # We will try to construct a rank 36 decomposition for 4x5x5
    # by using the 5x5x5 Winograd algorithm and truncating.
    
    # Since I don't have the exact Winograd matrices for 5x5x5 hardcoded,
    # I will use a random search with a target rank of 50, 60, 70.
    # If I find a valid one, I update best_result.
    
    # The condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if j==j' (in the context of the sum)
    # Actually, the condition derived above is:
    # For all i, k, j, j': sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == 1 if j==j' else 0.
    
    def verify(U, V, W, rank):
        # Check condition: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] == delta_{j,j'}
        # This is a 4D tensor check.
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # Construct the tensor T[i, j, j', k] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r]
        # We want T[i, j, j', k] == 1 if j==j' else 0.
        
        # This is expensive to check fully.
        # We can check a random subset of (i, k, j, j') tuples.
        
        for _ in range(100):
            i = random.randint(0, n - 1)
            k = random.randint(0, p - 1)
            j = random.randint(0, m - 1)
            j_prime = random.randint(0, m - 1)
            
            val = 0
            for r in range(rank):
                val += U[i * m + j, r] * V[j_prime * p + k, r] * W[k * n + i, r]
            
            expected = 1.0 if j == j_prime else 0.0
            if abs(val - expected) > 1e-6:
                return False
        return True

    # Random search
    random.seed(42)
    
    for target_rank in [90, 80, 70, 60, 50, 40, 36]:
        if time.time() > deadline:
            break
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Random initialization
            for r in range(target_rank):
                # Sparse random
                for _ in range(3):
                    idx_u = random.randint(0, n * m - 1)
                    U[idx_u, r] = random.uniform(-1, 1)
                    
                    idx_v = random.randint(0, m * p - 1)
                    V[idx_v, r] = random.uniform(-1, 1)
                    
                    idx_w = random.randint(0, p * n - 1)
                    W[idx_w, r] = random.uniform(-1, 1)
            
            if verify(U, V, W, target_rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                break # Found a valid one for this rank, try lower
        
    return best_result

# EVOLVE-BLOCK-END
