import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    max_time = 140.0  # Leave margin for 150s limit
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    nm, mp, pn = n * m, m * p, p * n

    def snap_and_verify(U, V, W, r):
        """Snap to nearest 0.5 and verify exactness."""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Check if all are half-integers (should be true after snap)
        # Verify exact decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U_s[c, u_idx] * V_s[c, v_idx] * W_s[c, w_idx]
                    if abs(s - 1.0) > 1e-9:
                        return None
        return {
            "U": U_s.tolist(),
            "V": V_s.tolist(),
            "W": W_s.tolist()
        }

    def compute_error_and_grad(params, r):
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp :]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        err = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Precompute S tensor
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = 0.0
                    for c in range(r):
                        s += U[c, u_idx] * V[c, v_idx] * W[c, w_idx]
                    S[i, j, k] = s
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    diff = S[i, j, k] - 1.0
                    err += diff * diff
                    for c in range(r):
                        grad_U[c, u_idx] += 2.0 * diff * V[c, v_idx] * W[c, w_idx]
                        grad_V[c, v_idx] += 2.0 * diff * U[c, u_idx] * W[c, w_idx]
                        grad_W[c, w_idx] += 2.0 * diff * U[c, u_idx] * V[c, v_idx]
                        
        return err, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Try ranks from 23 down to 18
    for r in range(rank_sb - 1, 17, -1):
        if time.time() - start_time > max_time:
            break
            
        total_vars = r * (nm + mp + pn)
        max_attempts = 2 if r >= 22 else 4
        
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            # Initialize with schoolbook-like structure plus noise
            x0 = np.zeros(total_vars)
            for c in range(min(r, rank_sb)):
                i, jk = divmod(c, m * p)
                j, k = divmod(jk, p)
                x0[c * nm + i * m + j] = 1.0
                x0[r * nm + c * mp + j * p + k] = 1.0
                x0[r * nm + r * mp + c * pn + k * n + i] = 1.0
            
            # Add small random noise to escape local minima
            if attempt > 0:
                noise_scale = 0.1 * attempt
                x0 += np.random.normal(0, noise_scale, total_vars)
            
            from scipy.optimize import minimize
            try:
                # Use L-BFGS-B for faster convergence
                res = minimize(compute_error_and_grad, x0, args=(r,), method='L-BFGS-B', 
                               jac=True, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                U_flat = res.x[:r * nm]
                V_flat = res.x[r * nm : r * nm + r * mp]
                W_flat = res.x[r * nm + r * mp :]
                
                U = U_flat.reshape(r, nm)
                V = V_flat.reshape(r, mp)
                W = W_flat.reshape(r, pn)
                
                result = snap_and_verify(U, V, W, r)
                if result is not None:
                    return result

    return {"U": U_sb, "V": V_sb, "W": W_sb}
