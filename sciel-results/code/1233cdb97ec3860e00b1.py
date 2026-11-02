"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)
    
    # Try structured block decompositions first
    # Split the 3x5x6 problem into smaller blocks
    
    # Strategy 1: Try ranks from a reasonable lower bound
    # Theoretical lower bound for 3x5x6 is at least 3*5*6/some_factor
    # Known results: 3x3x3 has rank 23, so 3x5x6 should be around 40-50
    
    # First, try a few structured approaches
    # Then do more targeted random search
    
    # Try ranks from 25 to 89 with increasing effort
    for r in range(25, 90):
        if time.time() > deadline - 5:
            break
        
        # Number of trials based on remaining time and rank
        remaining = deadline - time.time()
        if remaining < 10:
            break
            
        # More trials for lower ranks (harder but more valuable)
        num_trials = max(10, int(remaining * 20 / (r - 24)))
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            result = _try_structured_decomp(n, m, p, r, trial)
            if result is not None:
                if r < best_rank:
                    best_rank = r
                    best_result = result
                    # Found a better rank, try to improve further
                    if r <= 35:
                        # Continue searching for even better
                        continue
                    else:
                        # Good enough, return
                        return best_result
    
    return best_result

def _schoolbook(n, m, p, rank):
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

def _try_structured_decomp(n, m, p, r, trial_seed):
    """Try to find a structured decomposition of rank r."""
    random.seed(trial_seed * 12345 + r)
    
    # Use a more structured approach:
    # 1. Generate sparse U, V, W with patterns based on the matrix structure
    # 2. Use small integer/half-integer values
    
    vals = [0, 0.5, 1, -0.5, -1, 0.25, -0.25, 2, -2, 1.5, -1.5]
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Each column represents one "rank-1" term
    # For matmul: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Strategy: make columns sparse with structured patterns
    for c in range(r):
        # Choose non-zero positions based on structure
        # U has n*m = 15 rows, V has m*p = 30 rows, W has p*n = 18 rows
        
        # For U: pick 2-3 rows (i*m + j positions)
        u_count = random.randint(2, 3)
        u_indices = random.sample(range(n * m), u_count)
        for idx in u_indices:
            U[idx, c] = random.choice(vals)
        
        # For V: pick 3-5 rows (j*p + k positions)
        v_count = random.randint(3, 5)
        v_indices = random.sample(range(m * p), v_count)
        for idx in v_indices:
            V[idx, c] = random.choice(vals)
        
        # For W: pick 2-3 rows (k*n + i positions)
        w_count = random.randint(2, 3)
        w_indices = random.sample(range(p * n), w_count)
        for idx in w_indices:
            W[idx, c] = random.choice(vals)
    
    # Verify
    if _verify_fast(n, m, p, U, V, W, r):
        # Convert to lists for output
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _verify_fast(n, m, p, U, V, W, r):
    """Fast verification using vectorized operations."""
    # For each (i,j,k), check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    
    # Vectorized check
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j, :]
            for k in range(p):
                v_row = V[j * p + k, :]
                w_row = W[k * n + i, :]
                s = np.sum(u_row * v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
