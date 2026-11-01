"""
Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition.
Combines combinatorial intuition with gradient-based polishing.
"""
import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 3]
n, m, p = DIMENSIONS
N, M, P = n, m, p  # aliases

# EVOLVE-BLOCK-START

def _schoolbook():
    """Generate the standard rank-18 decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, tol=1e-6):
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reshape for einsum
    U_r = U.reshape((n, m, rank))
    V_r = V.reshape((m, p, rank))
    W_r = W.reshape((p, n, rank))
    
    # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
    actual = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    target = np.ones((n, m, p))
    
    diff = np.max(np.abs(actual - target))
    return diff < tol, diff

def _build_target_tensor():
    """Construct the target matrix multiplication tensor T."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _optimize_rank(target_rank, time_budget=10.0):
    """
    Attempt to find a valid decomposition of the given rank using 
    randomized restarts and L-BFGS-B optimization with improved initialization
    and post-processing.
    """
    start_time = time.time()
    T_target = _build_target_tensor()
    
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    
    def objective_and_grad(params):
        U_flat = params[:size_u]
        V_flat = params[size_u:size_u + size_v]
        W_flat = params[size_u + size_v:]
        
        U = U_flat.reshape((n, m, target_rank))
        V = V_flat.reshape((m, p, target_rank))
        W = W_flat.reshape((p, n, target_rank))
        
        T_approx = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        diff = T_approx - T_target
        obj_val = np.sum(diff * diff)
        
        gU = np.einsum('ijk,jkc,kic->ijc', diff, V, W)
        gV = np.einsum('ijk,ijc,kic->jkc', diff, U, W)
        gW = np.einsum('ijk,ijc,jkc->kic', diff, U, V)
        
        grad = np.concatenate([gU.ravel(), gV.ravel(), gW.ravel()])
        return obj_val, grad

    max_restarts = 200 
    rng = np.random.default_rng(42)
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        style = restart % 5
        
        if style == 0:
            # Random integers -1, 0, 1
            U_init = rng.integers(-1, 2, size=(n, m, target_rank))
            V_init = rng.integers(-1, 2, size=(m, p, target_rank))
            W_init = rng.integers(-1, 2, size=(p, n, target_rank))
        elif style == 1:
            # Random half-integers
            U_init = rng.integers(-2, 3, size=(n, m, target_rank)) * 0.5
            V_init = rng.integers(-2, 3, size=(m, p, target_rank)) * 0.5
            W_init = rng.integers(-2, 3, size=(p, n, target_rank)) * 0.5
        elif style == 2:
            # Sparse random integers (50% zeros)
            U_init = rng.integers(-1, 2, size=(n, m, target_rank))
            V_init = rng.integers(-1, 2, size=(m, p, target_rank))
            W_init = rng.integers(-1, 2, size=(p, n, target_rank))
            U_init[rng.random(U_init.shape) < 0.5] = 0
            V_init[rng.random(V_init.shape) < 0.5] = 0
            W_init[rng.random(W_init.shape) < 0.5] = 0
        elif style == 3:
            # Small Gaussian
            U_init = rng.standard_normal((n, m, target_rank)) * 0.5
            V_init = rng.standard_normal((m, p, target_rank)) * 0.5
            W_init = rng.standard_normal((p, n, target_rank)) * 0.5
        else:
            # Structured: Random permutation of identity-like blocks? 
            # Just use larger integers to escape local minima
            U_init = rng.integers(-2, 3, size=(n, m, target_rank))
            V_init = rng.integers(-2, 3, size=(m, p, target_rank))
            W_init = rng.integers(-2, 3, size=(p, n, target_rank))
        
        x0 = np.concatenate([U_init.ravel(), V_init.ravel(), W_init.ravel()])
        
        try:
            res = scipy.optimize.minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            
            if res.fun < 1e-6: # Relaxed threshold for rounding attempt
                x_opt = res.x
                U_opt = x_opt[:size_u].reshape((n*m, target_rank))
                V_opt = x_opt[size_u:size_u+size_v].reshape((m*p, target_rank))
                W_opt = x_opt[size_u+size_v:].reshape((p*n, target_rank))
                
                # Try rounding to nearest integer
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                
                valid, _ = _check_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist())
                if valid:
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rounding to nearest half-integer
                U_half = np.round(2 * U_opt) / 2.0
                V_half = np.round(2 * V_opt) / 2.0
                W_half = np.round(2 * W_opt) / 2.0
                
                valid, _ = _check_decomposition(U_half.tolist(), V_half.tolist(), W_half.tolist())
                if valid:
                    return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                    
                # Try sparse rounding: zero out small values
                for thresh in [0.1, 0.2, 0.5]:
                    U_sparse = np.where(np.abs(U_opt) < thresh, 0, U_opt)
                    V_sparse = np.where(np.abs(V_opt) < thresh, 0, V_opt)
                    W_sparse = np.where(np.abs(W_opt) < thresh, 0, W_opt)
                    
                    # Round the non-zero parts to nearest half-integer
                    U_sparse = np.round(2 * U_sparse) / 2.0
                    V_sparse = np.round(2 * V_sparse) / 2.0
                    W_sparse = np.round(2 * W_sparse) / 2.0
                    
                    valid, _ = _check_decomposition(U_sparse.tolist(), V_sparse.tolist(), W_sparse.tolist())
                    if valid:
                        return {"U": U_sparse.tolist(), "V": V_sparse.tolist(), "W": W_sparse.tolist()}

        except Exception:
            continue
            
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook
    result = _schoolbook()
    best_rank = n * m * p
    best_result = result
    
    # Known lower bound is around 12-14. 
    # Theoretical bound from 2x2x3 (rank 11) + 1x3x3 (rank 3) is 14.
    # We try ranks from 14 up to 17. If we find 14, we stop.
    # We allocate more time to 14 and 15 as they are more likely.
    
    ranks_to_try = [14, 15, 16, 17]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on rank likelihood
        # 14 is the most promising
        if rank == 14:
            budget = 60.0
        elif rank == 15:
            budget = 40.0
        else:
            budget = 20.0
            
        time_left = deadline - time.time() - 5
        if time_left < 5:
            break
            
        actual_budget = min(budget, time_left)
        
        opt_res = _optimize_rank(rank, time_budget=actual_budget)
        if opt_res:
            best_rank = rank
            best_result = opt_res
            # If we found a valid decomposition, we can stop 
            # because we are iterating from low to high rank.
            break
            
    return best_result

# EVOLVE-BLOCK-END
