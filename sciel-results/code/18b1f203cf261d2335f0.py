import numpy as np
import time
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    # Precompute index mappings for verification
    # U: (n*m, r), index i*m + j
    # V: (m*p, r), index j*p + k
    # W: (p*n, r), index k*n + i
    
    def verify_exact(U, V, W):
        """
        Verify if U, V, W (float arrays) represent a valid half-integer decomposition.
        Returns True if valid, False otherwise.
        """
        # Round to nearest half-integer
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        
        # Check if they are effectively integers (half-integers * 2)
        if not (np.allclose(U2, np.round(U2)) and np.allclose(V2, np.round(V2)) and np.allclose(W2, np.round(W2))):
            return False
            
        U2 = U2.astype(int)
        V2 = V2.astype(int)
        W2 = W2.astype(int)
        
        rank = U2.shape[1]
        
        # Check all 36 constraints: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1
        # With 2x scaling: sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r] = 8
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U2[i*m+j, r] * V2[j*p+k, r] * W2[k*n+i, r]
                    if s != 8:
                        return False
        return True

    def get_error(U, V, W):
        """Compute max absolute error for the decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def objective(params, n, m, p, rank):
        """
        Smooth surrogate objective using log-sum-exp of absolute errors
        plus a small L2 regularization to encourage stability.
        """
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        errors = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute dot product for this (i,j,k)
                    # U[i*m+j, :], V[j*p+k, :], W[k*n+i, :]
                    u_row = U[i*m+j, :]
                    v_row = V[j*p+k, :]
                    w_row = W[k*n+i, :]
                    s = np.sum(u_row * v_row * w_row)
                    errors[idx] = abs(s - 1.0)
                    idx += 1
        
        # Log-sum-exp for smooth max
        if np.max(errors) < 1e-12:
            soft_max = 0.0
        else:
            beta = 10.0
            shifted = beta * errors
            shifted_max = np.max(shifted)
            log_sum_exp = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
            soft_max = log_sum_exp / beta
            
        # Regularization: encourage smaller norms (helps with snapping)
        reg = 1e-5 * (np.sum(U_flat**2) + np.sum(V_flat**2) + np.sum(W_flat**2))
        
        return soft_max + reg

    def try_optimize(rank, seed, time_limit_per_attempt=2.0):
        """
        Attempt to find a decomposition of given rank using scipy.optimize.
        Returns (U, V, W, error) or None.
        """
        rng = np.random.default_rng(seed)
        
        # Sparse initialization to encourage "hypergraph" structure
        # Most entries are 0, some are random
        density = 0.2 
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        # Apply sparsity mask
        mask_U = (rng.random((n*m, rank)) < density)
        mask_V = (rng.random((m*p, rank)) < density)
        mask_W = (rng.random((p*n, rank)) < density)
        
        U = np.where(mask_U, U, 0)
        V = np.where(mask_V, V, 0)
        W = np.where(mask_W, W, 0)
        
        # Scale down to avoid overflow/instability initially
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        start_t = time.time()
        try:
            # Use L-BFGS-B for efficient gradient descent
            res = minimize(
                objective,
                params,
                args=(n, m, p, rank),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000,
                    'maxfun': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'iprint': 0
                }
            )
            
            if time.time() - start_t > time_limit_per_attempt:
                return None
            
            final_params = res.x
            U_opt = final_params[:n*m*rank].reshape(n*m, rank)
            V_opt = final_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = final_params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank].reshape(p*n, rank)
            
            err = get_error(U_opt, V_opt, W_opt)
            
            return U_opt, V_opt, W_opt, err
            
        except Exception as e:
            return None

    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = schoolbook()
    
    # Search for lower ranks
    # Known lower bounds for 3x3x4 are around 30-32? 
    # We start from a reasonable low rank and go up.
    # If we find a valid one, we update best and continue lower? 
    # No, we want the MINIMAL rank. So we should try low ranks first.
    # However, low ranks are harder to find.
    # Let's try from 28 up to 35.
    
    for rank in range(28, rank_best):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Try multiple seeds
        for seed in range(50):
            if time.time() > deadline - 5:
                break
                
            res = try_optimize(rank, seed)
            if res is None:
                continue
                
            U_opt, V_opt, W_opt, err = res
            
            if err < 1e-6:
                # Snap to half-integers
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                if verify_exact(U_snap, V_snap, W_snap):
                    U_best = U_snap
                    V_best = V_snap
                    W_best = W_snap
                    rank_best = rank
                    found = True
                    # If we found a valid decomposition, we can stop searching this rank
                    # and move to lower ranks
                    break
        
        # If we found a valid decomposition for this rank, we don't need to check higher ranks
        # because we are iterating upwards? 
        # Wait, I am iterating from 28 UP to 35.
        # If I find rank 30, I should stop and return 30? 
        # No, I want the MINIMAL. So if I start at 28 and find nothing, I try 29.
        # If I find 30, I should check if 29 or 28 is possible? 
        # The loop structure `for rank in range(28, rank_best)` goes 28, 29, ...
        # If I find a solution at 30, I should update best to 30.
        # But I haven't checked 28 or 29 yet if I started at 28? 
        # Yes, I did. If 28 and 29 failed, 30 is the best so far.
        # But maybe 28 was missed due to bad seed?
        # To be safe, if we find a solution, we should ideally verify if lower ranks are possible.
        # But given time constraints, finding *any* rank < 36 is a big win.
        # Let's just keep the best found.
        
        if found:
            # We found a valid decomposition at this rank.
            # Since we are going up, this is the lowest rank we've found so far 
            # that worked in this pass. 
            # We can break out of the rank loop if we want to ensure we don't waste time
            # on higher ranks, but we might have missed a lower one.
            # However, the probability of finding a lower one in remaining time is low 
            # if we already used 50 seeds on higher ones? 
            # Actually, we iterate 28, 29, 30...
            # If 28 fails, we try 29. If 29 fails, we try 30.
            # If 30 succeeds, we have a rank 30 solution.
            # We can stop searching for HIGHER ranks (31, 32...) because we want minimal.
            break

    # If no better than schoolbook was found, return schoolbook.
    # Convert to lists for the interface
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
