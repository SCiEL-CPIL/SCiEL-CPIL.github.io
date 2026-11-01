import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0

    # Schoolbook solution
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = r_sb

    # Target tensor
    T_target = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i, j, k] = 1.0

    def verify_numpy(U, V, W, R):
        try:
            U_t = U.reshape(n, m, R)
            V_t = V.reshape(m, p, R)
            W_t = W.reshape(p, n, R)
            T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
            return np.allclose(T, 1.0, atol=1e-9)
        except Exception:
            return False

    def loss_and_grad(params, R):
        # params is a flat array of size R*(nm + mp + pn)
        nm, mp, pn = n * m, m * p, p * n
        U_flat = params[:R * nm]
        V_flat = params[R * nm : R * nm + R * mp]
        W_flat = params[R * nm + R * mp : R * nm + R * mp + R * pn]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # Reconstruct T
        T = np.einsum('ijr,jkr,ki->ijk', U, V, W)
        diff = T - T_target
        loss = np.sum(diff * diff)
        
        # Gradients
        # dL/dU = 2 * einsum('ijk,jkr,ki->ijr', diff, V, W)
        # dL/dV = 2 * einsum('ijk,ijr,ki->jkr', diff, U, W)
        # dL/dW = 2 * einsum('ijk,ijr,jkr->ki', diff, U, V)
        
        gU = 2.0 * np.einsum('ijk,jkr,ki->ijr', diff, V, W)
        gV = 2.0 * np.einsum('ijk,ijr,ki->jkr', diff, U, W)
        gW = 2.0 * np.einsum('ijk,ijr,jkr->ki', diff, U, V)
        
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        return loss, grad

    # Try to find lower ranks
    # We'll try ranks from 47 down to 40.
    # For each rank, we try a few random initializations and run optimization.
    
    for target_rank in range(47, 39, -1):
        if time.time() > deadline - 10: # Leave 10s buffer
            break
            
        nm, mp, pn = n * m, m * p, p * n
        total_vars = target_rank * (nm + mp + pn)
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly
            params = np.random.randn(total_vars) * 0.5
            
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(
                    loss_and_grad, 
                    params, 
                    method='L-BFGS-B', 
                    jac=True,
                    options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                
                if res.success and res.fun < 1e-8:
                    # Extract optimized params
                    opt_params = res.x
                    U_opt = opt_params[:target_rank * nm].reshape(n, m, target_rank)
                    V_opt = opt_params[target_rank * nm : target_rank * nm + target_rank * mp].reshape(m, p, target_rank)
                    W_opt = opt_params[target_rank * nm + target_rank * mp :].reshape(p, n, target_rank)
                    
                    # Round to nearest half-integer
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify
                    if verify_numpy(U_rounded, V_rounded, W_rounded, target_rank):
                        best_result = {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        best_rank = target_rank
                        # Found a solution, can try lower ranks if time permits
                        break 
            except Exception:
                pass
        
        # If we found a solution at this rank, we can break or continue.
        # Continuing to lower ranks is good if we have time.
        # But if we found 47, trying 46 might take too long.
        # Let's just break if we found a solution to be safe, or continue if time is ample.
        if best_rank < r_sb:
             # If we found a lower rank, we might want to try to go even lower.
             # But L-BFGS is slow. Let's see time.
             if time.time() > deadline - 30:
                 break

    return best_result
