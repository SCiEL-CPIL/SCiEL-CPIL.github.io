import numpy as np
import time
import random
import sys

# Dimensions
n, m, p = 2, 3, 4

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute indices for the 24 constraints
    # Constraint for (i, j, k):
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    u_idxs = []
    v_idxs = []
    w_idxs = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
                
    u_idxs = np.array(u_idxs)
    v_idxs = np.array(v_idxs)
    w_idxs = np.array(w_idxs)
    
    def compute_error(U, V, W):
        """Compute sum of squared errors."""
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # We need sum over r of U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        # Vectorized:
        # Create arrays of shape (24, R)
        U_expanded = U[u_idxs, :] # (24, R)
        V_expanded = V[v_idxs, :] # (24, R)
        W_expanded = W[w_idxs, :] # (24, R)
        
        products = U_expanded * V_expanded * W_expanded # (24, R)
        sums = np.sum(products, axis=1) # (24,)
        errors = sums - 1.0
        return np.sum(errors**2)
        
    def verify_exact(U, V, W, tol=1e-9):
        return compute_error(U, V, W) < tol

    def round_to_halfint(X):
        return np.round(X * 2) / 2

    def integer_coordinate_descent(U, V, W, max_iters=5000, tol=1e-12):
        """
        Refine a half-integer solution using coordinate descent on the integer grid.
        U, V, W are assumed to be half-integers (or close to it).
        """
        U = round_to_halfint(U)
        V = round_to_halfint(V)
        W = round_to_halfint(W)
        
        current_err = compute_error(U, V, W)
        
        if current_err < tol:
            return U, V, W, True
            
        for _ in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            improved = False
            
            # Try updating U
            # For each non-zero element in U, try +/- 0.5
            # To be efficient, only try elements that are part of active constraints?
            # Actually, all elements in U are potentially active.
            # But U has size 6*R. R is ~20. So 120 elements.
            # V has 12*R = 240 elements.
            # W has 8*R = 160 elements.
            # Total ~520 elements.
            # For each, 3 choices (keep, -0.5, +0.5).
            # This is fast.
            
            # Flatten indices
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            
            # We'll iterate through all variables
            # To avoid O(N^2) in error calculation, we just recompute error fully.
            # Error calculation is O(24 * R) = O(480).
            # 520 variables * 480 ops = 250,000 ops per iteration.
            # 5000 iterations = 1.25 billion ops. Might be too slow in Python.
            
            # Optimization: Only check variables that are "important".
            # Or, just do a limited number of random variable updates per iteration.
            
            # Let's do random coordinate descent.
            num_vars = U.size + V.size + W.size
            # Sample a few variables to update
            num_updates = min(10, num_vars)
            
            for _ in range(num_updates):
                var_idx = random.randint(0, num_vars - 1)
                
                if var_idx < U.size:
                    u_flat_idx = var_idx
                    r, c = np.unravel_index(u_flat_idx, U.shape)
                    old_val = U[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        U[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-15:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            U[r, c] = old_val
                            
                elif var_idx < U.size + V.size:
                    v_flat_idx = var_idx - U.size
                    r, c = np.unravel_index(v_flat_idx, V.shape)
                    old_val = V[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        V[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-15:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            V[r, c] = old_val
                            
                else:
                    w_flat_idx = var_idx - U.size - V.size
                    r, c = np.unravel_index(w_flat_idx, W.shape)
                    old_val = W[r, c]
                    for delta in [-0.5, 0.5]:
                        new_val = old_val + delta
                        if new_val == old_val: continue
                        W[r, c] = new_val
                        new_err = compute_error(U, V, W)
                        if new_err < current_err - 1e-15:
                            current_err = new_err
                            improved = True
                            break
                        else:
                            W[r, c] = old_val
                            
            if not improved:
                # Try a larger step or break
                # If stuck, break
                break
                
            if current_err < tol:
                return U, V, W, True
                
        return U, V, W, False

    def try_rank_numeric(R, num_restarts=5, seed=0):
        """
        Try to find a rank-R solution using numerical optimization + integer refinement.
        """
        if time.time() > deadline - 20:
            return None
            
        for restart in range(num_restarts):
            if time.time() > deadline - 20:
                return None
                
            np.random.seed(seed + restart)
            
            # Initialize U, V, W
            # Strategy: Start with random small values
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            # Use scipy to minimize error
            # We need to flatten variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U = x[:U.size].reshape(n*m, R)
                V = x[U.size:U.size+V.size].reshape(m*p, R)
                W = x[U.size+V.size:].reshape(p*n, R)
                return compute_error(U, V, W)
                
            # Use L-BFGS-B
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10})
                x_opt = res.x
                U_opt = x_opt[:U.size].reshape(n*m, R)
                V_opt = x_opt[U.size:U.size+V.size].reshape(m*p, R)
                W_opt = x_opt[U.size+V.size:].reshape(p*n, R)
            except Exception:
                continue
                
            err_opt = compute_error(U_opt, V_opt, W_opt)
            
            # If error is small, try to round and refine
            if err_opt < 1.0: # Threshold for promising solution
                U_int = round_to_halfint(U_opt)
                V_int = round_to_halfint(V_opt)
                W_int = round_to_halfint(W_opt)
                
                U_ref, V_ref, W_ref, success = integer_coordinate_descent(U_int, V_int, W_int, max_iters=2000)
                
                if success:
                    return U_ref, V_ref, W_ref
                    
        return None

    # Try ranks from 23 down to 20
    # 24 is the fallback
    for R in [23, 22, 21, 20]:
        if time.time() > deadline - 30:
            break
            
        sol = try_rank_numeric(R, num_restarts=10, seed=R*100)
        if sol is not None:
            U, V, W = sol
            # Final verification
            if verify_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook rank 24
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
