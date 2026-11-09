"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions
    # Based on known results, try from 20 up to 44
    for target_rank in range(20, rank_school):
        if time.time() > deadline - 20:
            break
        
        success, U_r, V_r, W_r = _search_decomposition(n, m, p, target_rank, deadline)
        if success:
            best_rank = target_rank
            best_U, best_V, best_W = U_r, V_r, W_r
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}


def _compute_error(n, m, p, U, V, W):
    """Compute the maximum absolute error in the decomposition."""
    rank = len(U[0])
    max_err = 0.0
    
    # Check the 1-entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    # Check the 0-entries (sample a subset for speed)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k = random.randint(0, p-1)
        k2 = random.randint(0, p-1)
        if i == i2 and j == j2 and k == k2:
            continue
        s = 0.0
        for r in range(rank):
            s += U[i*m+j][r] * V[j2*p+k2][r] * W[k*n+i2][r]
        err = abs(s)
        if err > max_err:
            max_err = err
    
    return max_err


def _search_decomposition(n, m, p, target_rank, deadline):
    """Search for a half-integer decomposition of the given rank."""
    random.seed(42)
    
    value_choices = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2, 2.5, -2.5]
    
    for attempt in range(3):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        U = [[random.choice(value_choices) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice(value_choices) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice(value_choices) for _ in range(target_rank)] for _ in range(p * n)]
        
        current_error = _compute_error(n, m, p, U, V, W)
        
        # Simulated annealing style local search
        temperature = 1.0
        cooling = 0.9999
        
        for it in range(100000):
            if time.time() > deadline - 20:
                break
            if current_error < 1e-9:
                return True, U, V, W
            
            # Pick random entry to modify
            which = random.random()
            if which < 0.3:
                row = random.randint(0, n * m - 1)
                col = random.randint(0, target_rank - 1)
                old_val = U[row][col]
                new_val = random.choice(value_choices)
                U[row][col] = new_val
                new_error = _compute_error(n, m, p, U, V, W)
            elif which < 0.6:
                row = random.randint(0, m * p - 1)
                col = random.randint(0, target_rank - 1)
                old_val = V[row][col]
                new_val = random.choice(value_choices)
                V[row][col] = new_val
                new_error = _compute_error(n, m, p, U, V, W)
            else:
                row = random.randint(0, p * n - 1)
                col = random.randint(0, target_rank - 1)
                old_val = W[row][col]
                new_val = random.choice(value_choices)
                W[row][col] = new_val
                new_error = _compute_error(n, m, p, U, V, W)
            
            # Accept or reject
            if new_error < current_error:
                current_error = new_error
            else:
                # Reject with some probability (simulated annealing)
                delta = new_error - current_error
                if random.random() < (1.0 / (1.0 + delta * 100)):
                    current_error = new_error
                else:
                    # Revert
                    if which < 0.3:
                        U[row][col] = old_val
                    elif which < 0.6:
                        V[row][col] = old_val
                    else:
                        W[row][col] = old_val
            
            temperature *= cooling
        
        if current_error < 1e-9:
            return True, U, V, W
    
    return False, None, None

# EVOLVE-BLOCK-END
