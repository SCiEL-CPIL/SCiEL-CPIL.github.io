import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    timeout = 120.0
    
    n, m, p = DIMENSIONS
    
    def schoolbook():
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
    
    def compute_loss_and_grad(params, r):
        U_size = n * m * r
        V_size = m * p * r
        W_size = p * n * r
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        
        U = U_flat.reshape(n * m, r)
        V = V_flat.reshape(m * p, r)
        W = W_flat.reshape(p * n, r)
        
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T[i,j,k] = sum_s U[i,j,s] V[j,k,s] W[k,i,s]
        T = np.einsum('ijs,jks,kis->ijk', U_3d, V_3d, W_3d)
        
        loss = np.sum((T - 1.0)**2)
        
        # Gradients
        # dL/dU[i,j,s] = 2 * sum_k (T[i,j,k] - 1) * V[j,k,s] * W[k,i,s]
        # dL/dV[j,k,s] = 2 * sum_i (T[i,j,k] - 1) * U[i,j,s] * W[k,i,s]
        # dL/dW[k,i,s] = 2 * sum_j (T[i,j,k] - 1) * U[i,j,s] * V[j,k,s]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    diff = T[i, j, k] - 1.0
                    if abs(diff) < 1e-10: continue
                    
                    # Update grad_U[i, j, :] += 2 * diff * V[j, k, :] * W[k, i, :]
                    grad_U[i, j, :] += 2.0 * diff * (V_3d[j, k, :] * W_3d[k, i, :])
                    
                    # Update grad_V[j, k, :] += 2 * diff * U[i, j, :] * W[k, i, :]
                    grad_V[j, k, :] += 2.0 * diff * (U_3d[i, j, :] * W_3d[k, i, :])
                    
                    # Update grad_W[k, i, :] += 2 * diff * U[i, j, :] * V[j, k, :]
                    grad_W[k, i, :] += 2.0 * diff * (U_3d[i, j, :] * V_3d[j, k, :])
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank(r, max_restarts=5):
        for restart in range(max_restarts):
            if time.time() - start_time > timeout - 10:
                return None
            
            np.random.seed(42 + restart)
            # Initialize with small random values
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = opt.minimize(
                    lambda x: compute_loss_and_grad(x, r)[0], 
                    params_init, 
                    jac=lambda x: compute_loss_and_grad(x, r)[1],
                    method='L-BFGS-B', 
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < 1e-6:
                    U_size = n * m * r
                    V_size = m * p * r
                    W_size = p * n * r
                    
                    U_flat = res.x[:U_size]
                    V_flat = res.x[U_size:U_size + V_size]
                    W_flat = res.x[U_size + V_size:]
                    
                    U = U_flat.reshape(n * m, r)
                    V = V_flat.reshape(m * p, r)
                    W = W_flat.reshape(p * n, r)
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    
                    if max_err < 1e-6:
                        return {
                            "U": [[float(x) for x in row] for row in U_r],
                            "V": [[float(x) for x in row] for row in V_r],
                            "W": [[float(x) for x in row] for row in W_r]
                        }
            except Exception:
                pass
                
        return None

    # Try ranks from 71 down to 40
    # Use fewer restarts for higher ranks to save time, or just scan down.
    for r in range(71, 39, -1):
        if time.time() - start_time > timeout - 15:
            break
        
        result = try_rank(r, max_restarts=3)
        if result is not None:
            return result
    
    return schoolbook()
