"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time
import itertools
from scipy.optimize import minimize, least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_decomposition(rank_terms):
    """
    Build U, V, W factor matrices from a list of term specifications.
    Each term is (U_dict, V_dict, W_dict) where each dict maps (i,j) or (j,k) or (k,i) to a scalar.
    """
    n, m, p = DIMENSIONS
    R = len(rank_terms)
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), val in u_dict.items():
            U[i * m + j, r] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, r] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, r] = val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, atol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    R = U.shape[1]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    
    return np.allclose(T_recon, T_target, atol=atol)

def _strassen_2x2_terms():
    """
    Return the 7 Strassen terms for the 2x2 core (indices 0,1).
    """
    terms = []
    terms.append(({(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1},
                  {(0,0): 1, (1,1): 1}))
    terms.append(({(1,0): 1, (1,1): 1},
                  {(0,0): 1},
                  {(0,1): 1, (1,1): -1}))
    terms.append(({(0,0): 1},
                  {(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1}))
    terms.append(({(1,1): 1},
                  {(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1}))
    terms.append(({(0,0): 1, (0,1): 1},
                  {(1,1): 1},
                  {(0,0): -1, (1,0): 1}))
    terms.append(({(1,0): 1, (0,0): -1},
                  {(0,0): 1, (0,1): 1},
                  {(1,1): 1}))
    terms.append(({(0,1): 1, (1,1): -1},
                  {(1,0): 1, (1,1): 1},
                  {(0,0): 1}))
    return terms

def _border_terms_standard():
    """Standard border terms (at least one index is 2)."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    return terms

def _try_standard_decomposition():
    """Standard schoolbook: rank 27."""
    n, m, p = DIMENSIONS
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append(({(i,j): 1}, {(j,k): 1}, {(k,i): 1}))
    U, V, W = _build_decomposition(terms)
    return U, V, W

def _objective_and_jacobian(params, n, m, p, R):
    """
    Objective function for optimization.
    params: flattened vector of size 3 * n*m*p * R
    """
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    
    U_flat = params[:size_U]
    V_flat = params[size_U:size_U + size_V]
    W_flat = params[size_U + size_V:]
    
    U = U_flat.reshape(n, m, R)
    V = V_flat.reshape(m, p, R)
    W = W_flat.reshape(p, n, R)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    T_target = np.ones((n, m, p))
    
    residual = T_recon - T_target
    return residual

def _try_optimization_for_rank(R, n, m, p, deadline, initial_guess=None):
    """
    Try to find a decomposition of rank R using optimization.
    """
    if time.time() > deadline:
        return None
        
    size_U = n * m * R
    size_V = m * p * R
    size_W = p * n * R
    total_size = size_U + size_V + size_W
    
    if initial_guess is None:
        # Random initialization
        np.random.seed(42)
        params = np.random.normal(0, 1, total_size)
    else:
        params = initial_guess.copy()
        
    # We use least_squares on the residual
    # To handle scaling invariance, we can add a small regularization term
    # or just rely on the optimizer. least_squares minimizes sum(residual^2)
    
    def residual_func(params):
        return _objective_and_jacobian(params, n, m, p, R)
    
    # We might need to fix the scale. 
    # A common trick: fix the norm of the first row of U to be 1, etc.
    # But let's try standard least squares first.
    
    try:
        result = least_squares(
            residual_func, 
            params, 
            method='lm', # Levenberg-Marquardt, good for small problems
            max_nfev=10000
        )
        
        if time.time() > deadline:
            return None

        final_residual = np.linalg.norm(result.fun)
        
        # If the residual is small enough, we found a solution
        if final_residual < 1e-6:
            # Extract U, V, W
            U_flat = result.x[:size_U]
            V_flat = result.x[size_U:size_U + size_V]
            W_flat = result.x[size_U + size_V:]
            
            U = U_flat.reshape(n * m, R)
            V = V_flat.reshape(m * p, R)
            W = W_flat.reshape(p * n, R)
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p, atol=1e-5):
                # Try to snap to 0.5 increments if close
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p, atol=1e-5):
                    return U_snap, V_snap, W_snap
                else:
                    return U, V, W
                    
    except Exception as e:
        pass
        
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Strategy 1: Strassen core + standard border (rank 26)
    if time.time() < deadline:
        terms = _strassen_2x2_terms() + _border_terms_standard()
        U, V, W = _build_decomposition(terms)
        if _verify_decomposition(U, V, W, n, m, p):
            R = U.shape[1]
            if R < best_rank:
                best_rank = R
                best_result = (U, V, W)
    
    # Strategy 2: Try to find lower rank solutions (25, 24, 23) using optimization
    # We try ranks from 25 down to 23.
    # We use the rank-26 solution to initialize the lower rank attempts?
    # It's hard to project directly. We will use random restarts for lower ranks.
    
    target_ranks = [25, 24, 23]
    
    for R_target in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        found = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            res = _try_optimization_for_rank(R_target, n, m, p, deadline)
            if res is not None:
                U, V, W = res
                if _verify_decomposition(U, V, W, n, m, p, atol=1e-5):
                    if R_target < best_rank:
                        best_rank = R_target
                        best_result = (U, V, W)
                        found = True
                        break # Found a solution for this rank, move to next lower rank
        
        if found:
            # If we found a solution for this rank, we can stop trying this rank
            # and move to the next lower one if time permits
            continue
    
    # Strategy 3: Pure standard (rank 27) as fallback
    if best_result is None or best_rank > 27:
        if time.time() < deadline:
            U, V, W = _try_standard_decomposition()
            if _verify_decomposition(U, V, W, n, m, p):
                R = U.shape[1]
                if R < best_rank:
                    best_rank = R
                    best_result = (U, V, W)
    
    if best_result is None:
        U, V, W = _try_standard_decomposition()
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
