"""
Strategy: Numerical optimization to find a low-rank tensor decomposition.
Instead of hard-coding a potentially flawed manual construction, we search 
for a decomposition of rank R < 30 using alternating least squares (ALS) 
or gradient-based optimization. We initialize with random values and refine 
until the reconstruction error is negligible, then snap to rationals (integers 
or half-integers) and verify. If no exact low-rank solution is found, we fall 
back to a known valid schoolbook decomposition (rank 30).
"""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target: Find rank R such that T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Known lower bound for 2x3x5 is around 23-25. Schoolbook is 30.
    # We will try ranks from 25 down to 23. If we find an exact one, great.
    # If not, we return schoolbook (30) to ensure validity.
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_rank = n * m * p  # Schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Precompute indices for the tensor T
    # T is implicitly 1s for all i, j, k
    # We need to solve for U, V, W
    
    def try_rank(R, max_iters=500):
        """
        Attempt to find a rank-R decomposition using ALS.
        Returns (U, V, W) if found, else None.
        """
        # Initialize U, V, W with random small values
        # Symmetry breaking: fix one column of U to identity-like or specific values?
        # No, just random init is usually fine for ALS on this size.
        
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        # ALS: Fix two, solve for the third.
        # The equation is: sum_r U_{ij}^r V_{jk}^r W_{ki}^r = 1
        # This is a 3-way tensor completion problem.
        
        # To make ALS efficient, we can reshape.
        # For fixed r, let u_r be vector of size n*m, v_r size m*p, w_r size p*n.
        # The tensor T = sum_r u_r v_r^T w_r^T (with appropriate permutations).
        
        # Let's use a simpler approach: Gradient Descent on the error.
        # Error E = sum_{i,j,k} (sum_r U_{ij}^r V_{jk}^r W_{ki}^r - 1)^2
        
        # Since R is small (~25) and dimensions are small, we can just do random restarts with local search.
        # Or use scipy.optimize if available? The prompt says SciPy is available.
        
        # Let's use a randomized local search with refinement.
        # Actually, ALS is better.
        
        # Let's implement a simple ALS step.
        # We need to solve for one matrix given the others.
        # For example, solve for U:
        # sum_r U_{ij}^r (V_{jk}^r W_{ki}^r) = 1
        # This is not linear in U because V and W depend on r.
        # Wait, for a fixed r, the term is U_{ij}^r * (V_{jk}^r * W_{ki}^r).
        # The sum is over r.
        # So for each (i,j), we have:
        # sum_r U_{ij}^r * X_{ij}^r = 1, where X_{ij}^r = sum_k V_{jk}^r W_{ki}^r ? No.
        # The index k is summed in the tensor definition?
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # There is no sum over k in the term itself, k is a free index of T.
        # So for each (i,j,k), we have a linear equation in the variables.
        # But the variables are coupled across (i,j,k) via the same r.
        
        # This is a non-convex problem.
        
        # Alternative: Use the fact that we can "fold" the tensor.
        # Let's try a different optimization: 
        # Minimize || T - sum_r u_r v_r w_r ||_F^2
        # where u_r, v_r, w_r are vectors.
        
        # We will use scipy.optimize.minimize with a random initial guess.
        
        from scipy.optimize import minimize
        
        def flatten_params(params):
            # params is a flat array of length R * (n*m + m*p + p*n)
            R_loc = len(params) // (n*m + m*p + p*n)
            U_loc = params[:R_loc*n*m].reshape(R_loc, n*m).T
            V_loc = params[R_loc*n*m:R_loc*(n*m+m*p)].reshape(R_loc, m*p).T
            W_loc = params[R_loc*(n*m+m*p):].reshape(R_loc, p*n).T
            return U_loc, V_loc, W_loc

        def error_func(params):
            U_loc, V_loc, W_loc = flatten_params(params)
            # Compute tensor reconstruction
            # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We can compute this efficiently using einsum
            # T_recon = np.einsum('ir,jr,kr->ijk', U_loc, V_loc, W_loc) 
            # Wait, indices:
            # U index: i*m+j -> let's call it I
            # V index: j*p+k -> let's call it J
            # W index: k*n+i -> let's call it K
            # We need T[i,j,k].
            # U[I, r], V[J, r], W[K, r].
            # I = i*m+j, J = j*p+k, K = k*n+i.
            # This coupling makes einsum tricky directly on flattened indices.
            
            # Let's loop for clarity since dimensions are tiny.
            T_recon = np.zeros((n, m, p))
            for r in range(R_loc):
                # u_r is shape (n*m, 1) -> reshape to (n, m)
                u_r = U_loc[:, r].reshape(n, m)
                # v_r is shape (m*p, 1) -> reshape to (m, p)
                v_r = V_loc[:, r].reshape(m, p)
                # w_r is shape (p*n, 1) -> reshape to (p, n)
                w_r = W_loc[:, r].reshape(p, n)
                
                # T_recon[i,j,k] += u_r[i,j] * v_r[j,k] * w_r[k,i]
                # This is a rank-1 tensor in (i,j,k).
                # We can use np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
                T_recon += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            
            # Target is 1 everywhere
            err = np.sum((T_recon - 1.0)**2)
            return err

        # We need to search for a good initial guess.
        # Random initialization often gets stuck in local minima.
        # We will try several random restarts.
        
        best_err = float('inf')
        best_params = None
        
        num_restarts = 20  # Keep small to fit in time
        
        for _ in range(num_restarts):
            # Random init
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            # Flatten
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Optimize
            try:
                res = minimize(error_func, params0, method='BFGS', 
                               options={'maxiter': 100, 'gtol': 1e-8})
                if res.success and res.fun < best_err:
                    best_err = res.fun
                    best_params = res.x
            except Exception:
                continue
            
            if best_err < 1e-10:
                break
        
        if best_err < 1e-10 and best_params is not None:
            U_res, V_res, W_res = flatten_params(best_params)
            # Snap to integers or half-integers
            # The problem allows integer/half-integer entries.
            # We can try to round to nearest 0.5
            U_snap = np.round(U_res * 2) / 2
            V_snap = np.round(V_res * 2) / 2
            W_snap = np.round(W_res * 2) / 2
            
            # Verify exactness with snapped values
            T_check = np.zeros((n, m, p))
            for r in range(R):
                u_r = U_snap[:, r].reshape(n, m)
                v_r = V_snap[:, r].reshape(m, p)
                w_r = W_snap[:, r].reshape(p, n)
                T_check += np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
            
            if np.allclose(T_check, 1.0, atol=1e-9):
                return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from 25 down to 23
    for R in range(25, 22, -1):
        if time.time() > deadline:
            break
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook if no low-rank solution found
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
