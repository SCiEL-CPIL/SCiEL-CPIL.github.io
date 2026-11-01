import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Seed solution (Rank 24)
    rank_seed = n * m * p
    U_seed = np.zeros((n * m, rank_seed), dtype=np.float64)
    V_seed = np.zeros((m * p, rank_seed), dtype=np.float64)
    W_seed = np.zeros((p * n, rank_seed), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_seed[i * m + j, col] = 1
                V_seed[j * p + k, col] = 1
                W_seed[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_seed, V_seed, W_seed
    best_rank = rank_seed
    
    start_time = time.time()
    timeout = 110.0 # Leave 10s margin
    
    # Constraints: (u_idx, v_idx, w_idx)
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                constraints.append((u_idx, v_idx, w_idx))
                
    num_constraints = len(constraints)
    
    # Pre-compute indices for vectorization
    u_idxs = np.array([c[0] for c in constraints])
    v_idxs = np.array([c[1] for c in constraints])
    w_idxs = np.array([c[2] for c in constraints])

    def error_function(U, V, W):
        vals = np.sum(U[u_idxs, :] * V[v_idxs, :] * W[w_idxs, :], axis=1)
        return np.sum((vals - 1.0) ** 2)

    def try_rank(r, time_limit=5.0):
        if time.time() - start_time > timeout - 5.0:
            return None
        
        np.random.seed(42)
        # Try a few random restarts
        for attempt in range(5):
            if time.time() - start_time > timeout - 5.0:
                return None
            
            # Random initialization
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Flatten for scipy
            def objective(params):
                if time.time() - start_time > timeout - 5.0:
                    return np.inf
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r :].reshape(p*n, r)
                
                vals = np.sum(U_flat[u_idxs, :] * V_flat[v_idxs, :] * W_flat[w_idxs, :], axis=1)
                loss = np.sum((vals - 1.0) ** 2)
                
                # L1 penalty for sparsity
                sparsity = np.sum(np.abs(U_flat)) + np.sum(np.abs(V_flat)) + np.sum(np.abs(W_flat))
                return loss + 0.01 * sparsity

            def gradient(params):
                if time.time() - start_time > timeout - 5.0:
                    return np.zeros_like(params)
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r :].reshape(p*n, r)
                
                vals = np.sum(U_flat[u_idxs, :] * V_flat[v_idxs, :] * W_flat[w_idxs, :], axis=1)
                diffs = 2 * (vals - 1.0)
                
                dU = np.zeros_like(U_flat)
                dV = np.zeros_like(V_flat)
                dW = np.zeros_like(W_flat)
                
                np.add.at(dU, u_idxs, diffs[:, None] * V_flat[v_idxs, :] * W_flat[w_idxs, :])
                np.add.at(dV, v_idxs, diffs[:, None] * U_flat[u_idxs, :] * W_flat[w_idxs, :])
                np.add.at(dW, w_idxs, diffs[:, None] * U_flat[u_idxs, :] * V_flat[v_idxs, :])
                
                # Gradient of L1 term is sign
                dU += 0.01 * np.sign(U_flat)
                dV += 0.01 * np.sign(V_flat)
                dW += 0.01 * np.sign(W_flat)
                
                return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                               options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
                
                if res.success or res.fun < 1e-8:
                    U_opt = res.x[:n*m*r].reshape(n*m, r)
                    V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Verify exactness
                    if error_function(U_rounded, V_rounded, W_rounded) < 1e-10:
                        return U_rounded, V_rounded, W_rounded
            except Exception:
                pass
                
        return None

    # Try lower ranks
    for r in [23, 22]:
        sol = try_rank(r, time_limit=5.0)
        if sol:
            best_U, best_V, best_W = sol
            best_rank = r
            break
            
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
