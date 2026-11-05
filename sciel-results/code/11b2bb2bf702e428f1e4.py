import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds margin
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def check_exact(U, V, W):
        """Check if U, V, W form an exact decomposition with integer/half-integer entries."""
        rank = U.shape[1]
        # Check entries
        for mat in [U, V, W]:
            for val in mat:
                # Check if val is close to k/2
                closest = round(val * 2) / 2.0
                if abs(val - closest) > 1e-9:
                    return False
        # Check decomposition
        T_recon = np.tensordot(U, V, axes=1)
        T_recon = np.tensordot(T_recon, W, axes=1)
        # T_recon shape is (n*m, m*p, p*n)
        if np.allclose(T_recon, T, atol=1e-9):
            # Snap to exact half-integers
            U_snap = np.round(U * 2) / 2.0
            V_snap = np.round(V * 2) / 2.0
            W_snap = np.round(W * 2) / 2.0
            # Verify snap is exact
            T_recon_snap = np.tensordot(np.tensordot(U_snap, V_snap, axes=1), W_snap, axes=1)
            if np.allclose(T_recon_snap, T, atol=1e-9):
                return True
        return False

    def find_decomposition(target_rank):
        """Try to find a decomposition of the given rank."""
        # Random initialization
        for _ in range(100): # 100 restarts per rank
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Optimization
            def flatten_params(x):
                return x
            
            def objective(params):
                U_flat, V_flat, W_flat = params[:n*m*target_rank], params[n*m*target_rank:n*m*target_rank + m*p*target_rank], params[n*m*target_rank + m*p*target_rank:]
                U = U_flat.reshape(n * m, target_rank)
                V = V_flat.reshape(m * p, target_rank)
                W = W_flat.reshape(p * n, target_rank)
                
                T_recon = np.tensordot(np.tensordot(U, V, axes=1), W, axes=1)
                diff = T_recon - T
                return np.sum(diff**2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-10})
                if result.fun < 1e-6:
                    U_opt = result.x[:n*m*target_rank].reshape(n * m, target_rank)
                    V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m * p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p * n, target_rank)
                    
                    if check_exact(U_opt, V_opt, W_opt):
                        return U_opt, V_opt, W_opt
            except Exception:
                pass
        return None

    import scipy.optimize
    
    # Try ranks from 25 upwards
    for r in range(25, 35):
        if time.time() > deadline:
            break
        res = find_decomposition(r)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists for output
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
