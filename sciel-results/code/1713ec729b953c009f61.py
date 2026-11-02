"""
Matrix multiplication tensor decomposition for 3x3 x 3x4.

Strategy:
1. Start with the schoolbook rank-36 decomposition.
2. Use vectorized least squares to optimize for lower ranks.
3. Robustly snap to half-integers and verify.
"""

import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _verify(U, V, W, n, m, p):
    """Check if the factorization is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return U, V, W

def _error_function(params, n, m, p, R):
    """
    Calculate the error of the tensor decomposition using vectorized operations.
    params is a flattened array of U, V, W.
    """
    num_u = n * m * R
    num_v = m * p * R
    num_w = p * n * R
    
    U_flat = params[:num_u]
    V_flat = params[num_u:num_u+num_v]
    W_flat = params[num_u+num_v:num_u+num_v+num_w]
    
    U = U_flat.reshape(n*m, R)
    V = V_flat.reshape(m*p, R)
    W = W_flat.reshape(p*n, R)
    
    # Pre-compute indices for vectorization
    # We need E[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is a 3-way tensor contraction.
    
    # Construct index arrays for i, j, k
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    
    # Map to row indices for U, V, W
    u_rows = i_idx * m + j_idx  # Shape (n, m, p)
    v_rows = j_idx * p + k_idx  # Shape (n, m, p)
    w_rows = k_idx * n + i_idx  # Shape (n, m, p)
    
    # Gather the relevant rows
    # U[u_rows, r] for all r
    # We can use advanced indexing.
    # U_gathered shape: (n, m, p, R)
    U_g = U[u_rows, :]
    V_g = V[v_rows, :]
    W_g = W[w_rows, :]
    
    # Element-wise product and sum over R
    # Product shape: (n, m, p, R)
    prod = U_g * V_g * W_g
    # Sum over last axis
    errors = np.sum(prod, axis=-1) - 1.0
    
    return errors.flatten()

def _try_rank(R, n, m, p, U_init, V_init, W_init, time_limit=5.0):
    """
    Try to find a valid decomposition of rank R using local search.
    """
    start_time = time.time()
    
    # Flatten initial matrices
    U_flat = np.array(U_init, dtype=float).flatten()
    V_flat = np.array(V_init, dtype=float).flatten()
    W_flat = np.array(W_init, dtype=float).flatten()
    
    # If initial rank is larger, we need to truncate or randomize
    # We pad with zeros or random small values if too small
    if len(U_flat) > n*m*R:
        U_flat = U_flat[:n*m*R]
    elif len(U_flat) < n*m*R:
        U_flat = np.pad(U_flat, (0, n*m*R - len(U_flat)))
        
    if len(V_flat) > m*p*R:
        V_flat = V_flat[:m*p*R]
    elif len(V_flat) < m*p*R:
        V_flat = np.pad(V_flat, (0, m*p*R - len(V_flat)))
        
    if len(W_flat) > p*n*R:
        W_flat = W_flat[:p*n*R]
    elif len(W_flat) < p*n*R:
        W_flat = np.pad(W_flat, (0, p*n*R - len(W_flat)))
        
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    # Add small noise to escape local minima
    # The noise magnitude should be scaled with the problem
    noise_scale = 0.5
    x0 = x0 + np.random.normal(0, noise_scale, size=x0.size)
    
    try:
        result = least_squares(
            _error_function, 
            x0, 
            args=(n, m, p, R),
            method='lm', # Levenberg-Marquardt is good for this
            max_nfev=2000,
            xtol=1e-10,
            ftol=1e-10
        )
        
        if time.time() - start_time > time_limit:
            return None
            
        # Check if the solution is good enough
        if np.max(np.abs(result.fun)) < 1e-6:
            # Extract solutions
            num_u = n * m * R
            num_v = m * p * R
            num_w = p * n * R
            
            U_sol = result.x[:num_u].reshape(n*m, R)
            V_sol = result.x[num_u:num_u+num_v].reshape(m*p, R)
            W_sol = result.x[num_u+num_v:num_u+num_v+num_w].reshape(p*n, R)
            
            # Snap to half-integers and integers
            # Try half-integers first
            U_snapped = np.round(U_sol * 2) / 2
            V_snapped = np.round(V_sol * 2) / 2
            W_snapped = np.round(W_sol * 2) / 2
            
            U_list = U_snapped.tolist()
            V_list = V_snapped.tolist()
            W_list = W_snapped.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
            # Try integers
            U_int = np.round(U_sol)
            V_int = np.round(V_sol)
            W_int = np.round(W_sol)
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist(), n, m, p):
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
                
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110s out of 120s
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_factors = (U_school, V_school, W_school)
    best_rank = n * m * p
    
    # Verify the schoolbook solution
    if _verify(U_school, V_school, W_school, n, m, p):
        best_factors = (U_school, V_school, W_school)
        best_rank = n * m * p
    else:
        return {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to reduce rank
    # We iterate from low to high? No, high to low is safer to find *a* solution, 
    # but we want the *minimal* rank.
    # The rank is likely around 20-30. Let's try 25 to 35.
    # Actually, let's try a range of ranks.
    
    for target_rank in range(20, best_rank + 1):
        if time.time() > deadline - 10:
            break
            
        # Use schoolbook as initial guess (truncated/padded)
        # Note: Truncating schoolbook doesn't give a valid starting point for lower rank,
        # but it provides structure. Random restarts are better.
        
        found = False
        # Try multiple restarts for this rank
        for restart in range(10):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_rand = np.random.normal(0, 1, (n*m, target_rank))
            V_rand = np.random.normal(0, 1, (m*p, target_rank))
            W_rand = np.random.normal(0, 1, (p*n, target_rank))
            
            # Also try initializing with schoolbook if rank matches
            if target_rank == best_rank:
                U_init = U_school
                V_init = V_school
                W_init = W_school
            else:
                U_init = U_rand.tolist()
                V_init = V_rand.tolist()
                W_init = W_rand.tolist()
            
            res = _try_rank(target_rank, n, m, p, U_init, V_init, W_init, time_limit=2.0)
            if res is not None:
                best_factors = res
                best_rank = target_rank
                found = True
                break # Found a valid decomposition for this rank, try lower
                
    return {"U": best_factors[0], "V": best_factors[1], "W": best_factors[2]}

# EVOLVE-BLOCK-END
