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
        Checks full tensor identity: sum_r U[i,j1,r] V[j2,k,r] W[k,i,r] = delta(j1, j2)
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
        
        # Check all constraints:
        # For all i, j1, j2, k: sum_r U[i,j1,r] V[j2,k,r] W[k,i,r] = 1 if j1==j2 else 0
        # With 2x scaling: sum_r U2[i,j1,r] V2[j2,k,r] W2[k,i,r] = 8 if j1==j2 else 0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U2[i*m+j1, r] * V2[j2*p+k, r] * W2[k*n+i, r]
                        target = 8 if j1 == j2 else 0
                        if s != target:
                            return False
        return True

    def get_error(U, V, W):
        """Compute max absolute error for the decomposition."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U[i*m+j1, r] * V[j2*p+k, r] * W[k*n+i, r]
                        target = 1.0 if j1 == j2 else 0.0
                        err = abs(s - target)
                        if err > max_err:
                            max_err = err
        return max_err

    def objective(params, n, m, p, rank):
        """
        Smooth surrogate objective using log-sum-exp of squared errors
        plus a small L2 regularization.
        """
        U_flat = params[:n*m*rank]
        V_flat = params[n*m*rank : n*m*rank + m*p*rank]
        W_flat = params[n*m*rank + m*p*rank : n*m*rank + m*p*rank + p*n*rank]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        errors = np.zeros(n*m*m*p)
        idx = 0
        for i in range(n):
            for j1 in range(m):
                for j2 in range(m):
                    for k in range(p):
                        # Compute sum_r U[i,j1,r] V[j2,k,r] W[k,i,r]
                        u_row = U[i*m+j1, :]
                        v_row = V[j2*p+k, :]
                        w_row = W[k*n+i, :]
                        s = np.sum(u_row * v_row * w_row)
                        target = 1.0 if j1 == j2 else 0.0
                        errors[idx] = (s - target) ** 2
                        idx += 1
        
        # Log-sum-exp for smooth max of squared errors
        if np.max(errors) < 1e-12:
            soft_max = 0.0
        else:
            beta = 10.0
            shifted = beta * errors
            shifted_max = np.max(shifted)
            log_sum_exp = shifted_max + np.log(np.sum(np.exp(shifted - shifted_max)))
            soft_max = log_sum_exp / beta
            
        # Regularization: encourage smaller norms
        reg = 1e-5 * (np.sum(U_flat**2) + np.sum(V_flat**2) + np.sum(W_flat**2))
        
        return soft_max + reg

    def try_optimize(rank, seed, time_limit_per_attempt=2.0):
        """
        Attempt to find a decomposition of given rank using scipy.optimize.
        Returns (U, V, W, error) or None.
        """
        rng = np.random.default_rng(seed)
        
        # Initialize with random linear combinations of schoolbook basis
        # Schoolbook has rank n*m*p.
        # We select 'rank' random vectors from the space spanned by schoolbook basis.
        # This ensures the initial guess is "close" to a valid decomposition structure.
        
        # Generate random coefficients for each rank component
        # U, V, W will be sums of rank terms.
        # Let's construct U, V, W directly as random matrices, but scaled?
        # Better: Start with schoolbook, then project to lower rank?
        # Simple heuristic: Random initialization often fails.
        # Let's try random initialization with sparsity, but fix the objective.
        
        density = 0.2 
        U = rng.standard_normal((n*m, rank))
        V = rng.standard_normal((m*p, rank))
        W = rng.standard_normal((p*n, rank))
        
        mask_U = (rng.random((n*m, rank)) < density)
        mask_V = (rng.random((m*p, rank)) < density)
        mask_W = (rng.random((p*n, rank)) < density)
        
        U = np.where(mask_U, U, 0)
        V = np.where(mask_V, V, 0)
        W = np.where(mask_W, W, 0)
        
        U *= 0.5
        V *= 0.5
        W *= 0.5
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        start_t = time.time()
        try:
            res = minimize(
                objective,
                params,
                args=(n, m, p, rank),
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'maxfun': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-6,
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
    # We know 36 is valid.
    # Try 35, 34, ...
    for rank in range(35, 27, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for seed in range(20):
            if time.time() > deadline - 5:
                break
                
            res = try_optimize(rank, seed)
            if res is None:
                continue
                
            U_opt, V_opt, W_opt, err = res
            
            if err < 1e-6:
                U_snap = np.round(U_opt * 2) / 2.0
                V_snap = np.round(V_opt * 2) / 2.0
                W_snap = np.round(W_opt * 2) / 2.0
                
                if verify_exact(U_snap, V_snap, W_snap):
                    U_best = U_snap
                    V_best = V_snap
                    W_best = W_snap
                    rank_best = rank
                    found = True
                    break
        
        if found:
            break

    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
