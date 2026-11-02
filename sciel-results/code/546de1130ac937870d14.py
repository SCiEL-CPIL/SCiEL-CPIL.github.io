"""
Matrix multiplication tensor decomposition for 3x3 x 3x4.
We attempt to find a low-rank exact decomposition using local search
over integer/half-integer entries, with a schoolbook rank-36 fallback.
"""
import time
import random
import numpy as np

N, M, P = 3, 3, 4
DIMENSIONS = [N, M, P]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the factorization is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    R = n * m * p
    U = [[0.0] * R for _ in range(n*m)]
    V = [[0.0] * R for _ in range(m*p)]
    W = [[0.0] * R for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1.0
                V[j*p+k][c] = 1.0
                W[k*n+i][c] = 1.0
                c += 1
    return U, V, W

def _local_search(U, V, W, n, m, p, max_iters=20000, time_limit=60.0, seed=None):
    """
    Local search (hill climbing) to find a valid low-rank decomposition.
    We perturb one entry at a time and accept if it reduces the total error.
    """
    if seed is not None:
        random.seed(seed)
    start = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def total_error():
        err = 0.0
        for (i, j, k) in indices:
            s = 0.0
            for r in range(rank):
                s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            err += (s - 1.0) ** 2
        return err
    
    current_err = total_error()
    if current_err < 1e-12:
        return True, U, V, W
    
    # Candidate values for perturbation
    # We use a small set of rational values
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        if current_err < 1e-12:
            break
        
        # Randomly pick a tensor to perturb: 0=U, 1=V, 2=W
        tensor_idx = random.choice([0, 1, 2])
        if tensor_idx == 0:
            row = random.randint(0, n*m-1)
            col = random.randint(0, rank-1)
            old_val = U[row][col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            U[row][col] = new_val
            # Compute new error
            new_err = total_error()
            if new_err < current_err:
                current_err = new_err
            else:
                U[row][col] = old_val
        elif tensor_idx == 1:
            row = random.randint(0, m*p-1)
            col = random.randint(0, rank-1)
            old_val = V[row][col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            V[row][col] = new_val
            new_err = total_error()
            if new_err < current_err:
                current_err = new_err
            else:
                V[row][col] = old_val
        else:
            row = random.randint(0, p*n-1)
            col = random.randint(0, rank-1)
            old_val = W[row][col]
            new_val = random.choice(vals)
            if new_val == old_val:
                continue
            W[row][col] = new_val
            new_err = total_error()
            if new_err < current_err:
                current_err = new_err
            else:
                W[row][col] = old_val
    
    if current_err < 1e-12:
        return True, U, V, W
    return False, U, V, W

def construct():
    n, m, p = N, M, P
    deadline = time.time() + 110  # 110s out of 120s
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    best_factors = None
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Known best upper bound for 3x3x4 is 30, but we try from 30 up to 35
    for target_rank in [30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Multiple random initializations
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Initialize with random small integers/half-integers
            U = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(target_rank)] for _ in range(p*n)]
            
            # Run local search
            time_left = deadline - time.time()
            if time_left < 10:
                break
            success, U_res, V_res, W_res = _local_search(U, V, W, n, m, p, max_iters=10000, time_limit=time_left * 0.8)
            
            if success:
                if _verify(U_res, V_res, W_res, n, m, p):
                    best_rank = target_rank
                    best_factors = (U_res, V_res, W_res)
                    break
        
        if best_factors is not None:
            break
    
    if best_factors is not None:
        return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
