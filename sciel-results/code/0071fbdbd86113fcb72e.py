import numpy as np
import time
import sys
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Build target tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
        
    # Flatten T for faster access in gradient
    T_flat = T.flatten()
    
    def objective_and_grad(params, r):
        """
        Objective function and gradient for minimizing ||T - U V W||^2.
        params: flattened [U, V, W] of shape (nm*r, mp*r, pn*r)
        """
        # Reshape
        U = params[:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r + mp*r].reshape(mp, r)
        W = params[nm*r + mp*r:].reshape(pn, r)
        
        # Compute reconstruction
        # R_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
        # Using einsum
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < 1e-10:
            return loss, np.zeros_like(params)
            
        # Compute gradients
        # dL/dU_{ir} = -2 sum_{j,k} R_{ijk} V_{jr} W_{kr}
        # dL/dV_{jr} = -2 sum_{i,k} R_{ijk} U_{ir} W_{kr}
        # dL/dW_{kr} = -2 sum_{i,j} R_{ijk} U_{ir} V_{jr}
        
        grad_U = -2 * np.einsum('ijk,jr,kr->ir', R, V, W)
        grad_V = -2 * np.einsum('ijk,ir,kr->jr', R, U, W)
        grad_W = -2 * np.einsum('ijk,ir,jr->kr', R, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return loss, grad

    # Schoolbook fallback
    best_rank = nm * mp * pn
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    
    np.random.seed(42)
    
    lower_bound = max(nm, mp, pn)
    
    # We will try ranks from lower_bound to best_rank - 1
    # To save time, we can jump to known good ranks if we have them, but let's try sequentially.
    # However, 30 to 59 is 30 ranks. If each takes 1 second, that's 30 seconds.
    # If each takes 10 seconds, that's 300 seconds (too long).
    # We need to be faster.
    
    # Optimization:
    # 1. Limit restarts per rank to 5.
    # 2. Limit iterations in minimize to 50.
    # 3. Use BFGS which is faster per step than L-BFGS-B for small problems, but L-BFGS-B is better for large.
    #    Here params ~ 1500-3000. L-BFGS-B is good.
    
    max_restarts = 5
    max_iter = 100
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Scale: 1/sqrt(rank) is often good
            scale = 1.0 / np.sqrt(rank)
            U_init = np.random.randn(nm, rank) * scale
            V_init = np.random.randn(mp, rank) * scale
            W_init = np.random.randn(pn, rank) * scale
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy minimize
            try:
                res = minimize(
                    objective_and_grad, 
                    params_init, 
                    args=(rank,), 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-8:
                    # Verify
                    U_opt = res.x[:nm*rank].reshape(nm, rank)
                    V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    recon = np.einsum('ir,jr,kr->ijk', U_opt, V_opt, W_opt)
                    if np.allclose(recon, T, atol=1e-6):
                        found = True
                        best_rank = rank
                        best_result = (U_opt, V_opt, W_opt)
                        break
                        
            except Exception:
                pass
                
        if found:
            break
            
    U_final, V_final, W_final = best_result
    
    # Convert to lists of floats as required by interface
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
