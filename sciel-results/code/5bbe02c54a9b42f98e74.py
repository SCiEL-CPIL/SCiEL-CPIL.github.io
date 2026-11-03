import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s
    
    # The rank-17 solution
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))
    
    U17[0, 0] = 1; U17[4, 0] = 1
    V17[0, 0] = 1; V17[4, 0] = 1
    W17[0, 0] = 1; W17[3, 0] = 1
    
    U17[3, 1] = 1; U17[4, 1] = 1
    V17[0, 1] = 1
    W17[1, 1] = 1; W17[3, 1] = -1
    
    U17[0, 2] = 1
    V17[1, 2] = 1; V17[4, 2] = -1
    W17[2, 2] = 1; W17[3, 2] = 1
    
    U17[4, 3] = 1
    V17[3, 3] = 1; V17[0, 3] = -1
    W17[0, 3] = 1; W17[1, 3] = 1
    
    U17[0, 4] = 1; U17[1, 4] = 1
    V17[4, 4] = 1
    W17[0, 4] = -1; W17[2, 4] = 1
    
    U17[3, 5] = 1; U17[0, 5] = -1
    V17[0, 5] = 1; V17[1, 5] = 1
    W17[3, 5] = 1
    
    U17[1, 6] = 1; U17[4, 6] = -1
    V17[3, 6] = 1; V17[4, 6] = 1
    W17[0, 6] = 1
    
    U17[2, 7] = 1; V17[6, 7] = 1; W17[0, 7] = 1
    U17[2, 8] = 1; V17[7, 8] = 1; W17[2, 8] = 1
    U17[5, 9] = 1; V17[6, 9] = 1; W17[1, 9] = 1
    U17[5, 10] = 1; V17[7, 10] = 1; W17[3, 10] = 1
    
    U17[0, 11] = 1; V17[2, 11] = 1; W17[4, 11] = 1
    U17[1, 12] = 1; V17[5, 12] = 1; W17[4, 12] = 1
    U17[2, 13] = 1; V17[8, 13] = 1; W17[4, 13] = 1
    U17[3, 14] = 1; V17[2, 14] = 1; W17[5, 14] = 1
    U17[4, 15] = 1; V17[5, 15] = 1; W17[5, 15] = 1
    U17[5, 16] = 1; V17[8, 16] = 1; W17[5, 16] = 1
    
    # Precompute index tuples for efficient residual/gradient
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                indices.append((u_idx, v_idx, w_idx))
    indices = np.array(indices)
    
    def residual(U, V, W):
        """Compute the residual for the tensor decomposition."""
        vals = np.einsum('ir,jr,kr->ijk', U[indices[:,0]], V[indices[:,1]], W[indices[:,2]])
        return np.sum((vals - 1.0) ** 2)
    
    def residual_and_grad(U, V, W):
        """Compute residual and gradients analytically."""
        vals = np.einsum('ir,jr,kr->ijk', U[indices[:,0]], V[indices[:,1]], W[indices[:,2]])
        err = vals - 1.0
        res = np.sum(err ** 2)
        
        # Gradient: d/dU[a,r] = 2 * sum over (i,j,k) where u_idx=i*m+j=a of err[i,j,k]*V[j*p+k,r]*W[k*n+i,r]
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for idx in range(len(indices)):
            u_i, v_i, w_i = indices[idx]
            e = 2 * err[idx // (m*p), (idx % (m*p)) // p, idx % p]
            # Actually let me redo this more carefully
            pass
        
        # More efficient: vectorized
        # For each (i,j,k) in indices, err[i,j,k] contributes to:
        # gU[u_idx, r] += 2*err[i,j,k]*V[v_idx,r]*W[w_idx,r]
        # gV[v_idx, r] += 2*err[i,j,k]*U[u_idx,r]*W[w_idx,r]
        # gW[w_idx, r] += 2*err[i,j,k]*U[u_idx,r]*V[v_idx,r]
        
        # err shape: (n, m, p) but we only use specific (i,j,k) entries
        # Let's reshape err to match indices
        err_flat = err.reshape(-1)  # This won't work directly since err is full n*m*p
        
        # Better approach: iterate
        for t in range(len(indices)):
            u_i, v_i, w_i = indices[t]
            i = u_i // m
            j = u_i % m
            k = v_i % p
            e = 2.0 * err[i, j, k]
            gU[u_i, :] += e * V[v_i, :] * W[w_i, :]
            gV[v_i, :] += e * U[u_i, :] * W[w_i, :]
            gW[w_i, :] += e * U[u_i, :] * V[v_i, :]
        
        return res, np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    # Try to find rank-16 solution
    R = 16
    best_res = residual(U17, V17, W17)
    best_solution = (U17.copy(), V17.copy(), W17.copy())
    
    rng = np.random.default_rng(42)
    
    # Try each column removal with multiple restarts
    for remove_col in range(17):
        if time.time() > deadline:
            break
        
        cols = [r for r in range(17) if r != remove_col]
        U0 = U17[:, cols].copy()
        V0 = V17[:, cols].copy()
        W0 = W17[:, cols].copy()
        
        # Multiple restarts: original + perturbed
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                U_start = U0.copy()
                V_start = V0.copy()
                W_start = W0.copy()
            else:
                # Add small random perturbation to escape local minima
                scale = 0.1 / restart
                U_start = U0 + scale * rng.standard_normal(U0.shape)
                V_start = V0 + scale * rng.standard_normal(V0.shape)
                W_start = W0 + scale * rng.standard_normal(W0.shape)
            
            x0 = np.concatenate([U_start.flatten(), V_start.flatten(), W_start.flatten()])
            U_size = U0.size
            V_size = V0.size
            
            def objective(x):
                U = x[:U_size].reshape(U0.shape)
                V = x[U_size:U_size+V_size].reshape(V0.shape)
                W = x[U_size+V_size:].reshape(W0.shape)
                return residual(U, V, W)
            
            def objective_grad(x):
                U = x[:U_size].reshape(U0.shape)
                V = x[U_size:U_size+V_size].reshape(V0.shape)
                W = x[U_size+V_size:].reshape(W0.shape)
                return residual_and_grad(U, V, W)
            
            # Use L-BFGS-B with analytical gradient
            try:
                result = minimize(objective, x0, jac=objective_grad, method='L-BFGS-B',
                                  options={'maxiter': 300, 'maxfun': 5000, 'gtol': 1e-8})
                
                if result.fun < best_res:
                    best_res = result.fun
                    U_best = result.x[:U_size].reshape(U0.shape)
                    V_best = result.x[U_size:U_size+V_size].reshape(V0.shape)
                    W_best = result.x[U_size+V_size:].reshape(W0.shape)
                    best_solution = (U_best.copy(), V_best.copy(), W_best.copy())
                    
                    # If close to zero, try to round to half-integers
                    if best_res < 1e-6:
                        U_round = np.round(U_best * 2) / 2
                        V_round = np.round(V_best * 2) / 2
                        W_round = np.round(W_best * 2) / 2
                        
                        res_round = residual(U_round, V_round, W_round)
                        if res_round < 1e-9:
                            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                            
            except Exception:
                pass

    # Fall back to rank-17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
