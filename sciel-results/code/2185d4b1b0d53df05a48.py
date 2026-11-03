import numpy as np
import time
import scipy.optimize

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    
    def validate(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        r = U.shape[1]
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    if j*p <= b < (j+1)*p:
                        k = b - j*p
                        if c == k*n + i:
                            continue
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val) > tol:
                        return False
        return True

    def get_tensor_indices():
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i*m+j, j*p+k, k*n+i
                    indices.append((a, b, c, 1.0))
        return indices

    def get_zero_indices():
        indices = []
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i, j = divmod(a, m)
                    if j*p <= b < (j+1)*p:
                        k = b - j*p
                        if c == k*n + i:
                            continue
                    indices.append((a, b, c, 0.0))
        return indices

    target_indices = get_tensor_indices()
    zero_indices = get_zero_indices()
    all_indices = target_indices + zero_indices

    def objective_and_grad(x):
        r = x.shape[0] // 3
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:2*n*m*r].reshape(m*p, r)
        W = x[2*n*m*r:].reshape(p*n, r)
        
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Use log-sum-exp style smooth max for better conditioning
        # Sum of squares with weighting: target terms weighted more heavily
        for a, b, c, t in target_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - t
            obj += 10.0 * err * err
            grad_U[a, :] += 20.0 * err * V[b, :] * W[c, :]
            grad_V[b, :] += 20.0 * err * U[a, :] * W[c, :]
            grad_W[c, :] += 20.0 * err * U[a, :] * V[b, :]
            
        # Zero terms with lighter weight
        for a, b, c, t in zero_indices:
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            err = val - t
            obj += err * err
            grad_U[a, :] += 2 * err * V[b, :] * W[c, :]
            grad_V[b, :] += 2 * err * U[a, :] * W[c, :]
            grad_W[c, :] += 2 * err * U[a, :] * V[b, :]
            
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad

    def polish(x_opt, U, V, W, r):
        """High-precision Newton-style polish for the last few iterations"""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for iteration in range(20):
            if time.time() > deadline:
                break
            # Compute residuals
            resid_U = np.zeros_like(U)
            resid_V = np.zeros_like(V)
            resid_W = np.zeros_like(W)
            
            # Jacobian-based Newton step: solve J^T J dx = -J^T r
            # For tensor decomposition, this is the Gauss-Newton system
            # We'll use a simplified approach: iterate with damped Newton
            
            # Compute gradient
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for a, b, c, t in all_indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err = val - t
                weight = 10.0 if t != 0 else 1.0
                grad_U[a, :] += 2 * weight * err * V[b, :] * W[c, :]
                grad_V[b, :] += 2 * weight * err * U[a, :] * W[c, :]
                grad_W[c, :] += 2 * weight * err * U[a, :] * V[b, :]
            
            # Check convergence
            grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
            if grad_norm < 1e-10:
                break
            
            # Damped Newton step: use line search
            alpha = 1.0
            for ls in range(10):
                U_new = U - alpha * grad_U
                V_new = V - alpha * grad_V
                W_new = W - alpha * grad_W
                
                # Compute objective
                obj_new = 0.0
                for a, b, c, t in all_indices:
                    val = np.dot(U_new[a, :], V_new[b, :] * W_new[c, :])
                    err = val - t
                    weight = 10.0 if t != 0 else 1.0
                    obj_new += weight * err * err
                
                if obj_new < 1e-12:
                    U, V, W = U_new, V_new, W_new
                    break
                elif obj_new < 0.5:
                    U, V, W = U_new, V_new, W_new
                    break
                else:
                    alpha *= 0.5
            
            if alpha < 1e-6:
                break
        
        return U, V, W

    # Try to find rank 28 decomposition
    if time.time() < deadline:
        r_target = 28
        size_U = n * m * r_target
        size_V = m * p * r_target
        size_W = p * n * r_target
        total_size = size_U + size_V + size_W
        
        best_solution = None
        best_obj = float('inf')
        
        for seed in range(80):
            if time.time() > deadline:
                break
            np.random.seed(seed)
            
            # Use multiple initialization strategies
            if seed % 3 == 0:
                x0 = np.random.randn(total_size) * 0.1
            elif seed % 3 == 1:
                x0 = np.random.randn(total_size) * 0.5
            else:
                x0 = np.random.randn(total_size) * 1.0
            
            try:
                res = scipy.optimize.minimize(
                    objective_and_grad, 
                    x0, 
                    jac=True, 
                    method='L-BFGS-B', 
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.fun < best_obj:
                    best_obj = res.fun
                    x_opt = res.x
                    U = x_opt[:size_U].reshape(n*m, r_target)
                    V = x_opt[size_U:size_U+size_V].reshape(m*p, r_target)
                    W = x_opt[size_U+size_V:].reshape(p*n, r_target)
                    
                    # Apply polish
                    U, V, W = polish(x_opt, U, V, W, r_target)
                    
                    if validate(U, V, W, tol=1e-8):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    elif validate(U, V, W, tol=1e-6):
                        best_solution = (U, V, W)
                        
            except Exception:
                pass
        
        # Return best found solution if it's close
        if best_solution is not None:
            U, V, W = best_solution
            if validate(U, V, W, tol=1e-5):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U_sb, "V": V_sb, "W": W_sb}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
