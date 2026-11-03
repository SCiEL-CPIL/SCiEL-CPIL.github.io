"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import random
import math
import sys
import time
from collections import defaultdict

def _validate(U, V, W, n, m, p, tol=1e-9):
    """
    Check if U, V, W exactly decompose the matrix multiplication tensor.
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank).
    The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_random_decomposition(n, m, p, target_rank, num_trials=10, time_limit=10):
    """
    Try to find a rank-`target_rank` decomposition using random search with local improvements.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random sparse matrices
        # Use a mix of zeros and small integers/half-integers
        U = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(n * m)]
        V = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(m * p)]
        W = [[random.choice([0, 0, 0, 1, -1, 0.5, -0.5]) for _ in range(target_rank)] for _ in range(p * n)]
        
        # Check if this random guess works (extremely unlikely)
        if _validate(U, V, W, n, m, p):
            return U, V, W
        
        # Local search: try to fix errors by adjusting entries
        # Compute current errors
        errors = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(target_rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    errors[(i, j, k)] = s - 1.0
        
        # Try local improvements for a few iterations
        max_iter = 50
        for it in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Find the largest error
            max_err_key = None
            max_err_val = 0.0
            for key, err in errors.items():
                if abs(err) > abs(max_err_val):
                    max_err_val = err
                    max_err_key = key
            
            if abs(max_err_val) < 1e-9:
                return U, V, W
            
            i, j, k = max_err_key
            # Try to adjust a random entry in U, V, or W that affects this (i,j,k)
            # U[i*m+j, c] affects (i,j,k) for all k
            # V[j*p+k, c] affects (i,j,k) for all i
            # W[k*n+i, c] affects (i,j,k) for all j
            
            # Pick a random column c
            c = random.randint(0, target_rank - 1)
            
            # Try adjusting U[i*m+j, c]
            current_u = U[i * m + j][c]
            # We want to reduce the error. 
            # Error contribution from this term: current_u * V[j*p+k, c] * W[k*n+i, c]
            # Let delta be the change in U. New error = old_error + delta * V * W
            # We want delta * V * W = -error
            v_val = V[j * p + k][c]
            w_val = W[k * n + i][c]
            prod = v_val * w_val
            if abs(prod) > 1e-9:
                delta = -max_err_val / prod
                # Round delta to nearest half-integer or integer to keep entries simple
                # But this might not work. Let's just try adding a small random perturbation.
                # Actually, let's try setting the value to reduce the error significantly.
                # New value = current_u + delta
                new_u = current_u + delta
                # Quantize to nearest 0.5
                new_u = round(new_u * 2) / 2
                
                # Apply and check if it improves the total error
                old_total_err = sum(abs(e) for e in errors.values())
                U[i * m + j][c] = new_u
                
                # Recompute errors affected by U[i*m+j, c]
                # This affects all (i, j, k') for k' in range(p)
                for k_prime in range(p):
                    s = 0.0
                    for c_prime in range(target_rank):
                        s += U[i * m + j][c_prime] * V[j * p + k_prime][c_prime] * W[k_prime * n + i][c_prime]
                    errors[(i, j, k_prime)] = s - 1.0
                
                new_total_err = sum(abs(e) for e in errors.values())
                if new_total_err >= old_total_err:
                    # Revert
                    U[i * m + j][c] = current_u
                    for k_prime in range(p):
                        s = 0.0
                        for c_prime in range(target_rank):
                            s += U[i * m + j][c_prime] * V[j * p + k_prime][c_prime] * W[k_prime * n + i][c_prime]
                        errors[(i, j, k_prime)] = s - 1.0
                else:
                    # Keep the change
                    pass
        
        # After local search, check again
        if _validate(U, V, W, n, m, p):
            return U, V, W
    
    return None

def _schoolbook_decomposition(n, m, p):
    """Return the standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 75
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    random.seed(123)
    
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # But for 3x5x5, the rank is likely higher.
    # Let's try from 40 up to 74.
    
    for target_rank in range(40, rank_school):
        result = _try_random_decomposition(n, m, p, target_rank, num_trials=5, time_limit=8)
        if result is not None:
            U, V, W = result
            if _validate(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
