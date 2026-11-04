import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, 50s margin
    
    # Target tensor
    T = np.ones((n, m, p))
    
    # Try ranks from low to high
    # Known upper bound is 24.
    # Try 20, 19, 18, 17, 16
    ranks_to_try = [20, 19, 18, 17, 16]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few restarts for this rank
        # Use more restarts for lower ranks as they are harder
        n_restarts = 20 if r >= 19 else 50
        
        for attempt in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Flatten variables
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Residual function
            def residuals(x):
                U_flat, V_flat, W_flat = x[:n*m*r], x[n*m*r:n*m*r+m*p*r], x[n*m*r+m*p*r:]
                U_mat = U_flat.reshape(n * m, r)
                V_mat = V_flat.reshape(m * p, r)
                W_mat = W_flat.reshape(p * n, r)
                
                # Compute T_rec
                T_rec = np.zeros((n, m, p))
                for r_idx in range(r):
                    # u is (n, m), v is (m, p), w is (p, n)
                    u = U_mat[:, r_idx].reshape(n, m)
                    v = V_mat[:, r_idx].reshape(m, p)
                    w = W_mat[:, r_idx].reshape(p, n)
                    
                    # T_rec[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                    # We can do this with einsum or loops
                    # u[i, j] * v[j, k] is (n, p)
                    uv = np.einsum('ij,jk->ik', u, v)
                    # uv[i, k] * w[k, i] is scalar? No.
                    # We need T_rec[i, j, k].
                    # The term is u[i, j] * v[j, k] * w[k, i].
                    # This is a 3-way tensor.
                    # We can compute it as:
                    # tmp[i, j, k] = u[i, j] * v[j, k] * w[k, i]
                    # This is not a standard matrix product.
                    # Let's use loops for clarity and speed for small n, m, p.
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_rec[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                                
                return (T - T_rec).flatten()
            
            # Solve
            try:
                result = least_squares(residuals, x0, method='trf', max_nfev=1000, xtol=1e-12, ftol=1e-12)
            except Exception:
                continue
                
            if result.success or np.linalg.norm(result.fun) < 1e-8:
                # Extract U, V, W
                U_sol = result.x[:n*m*r].reshape(n * m, r)
                V_sol = result.x[n*m*r:n*m*r+m*p*r].reshape(m * p, r)
                W_sol = result.x[n*m*r+m*p*r:].reshape(p * n, r)
                
                # Snap to half-integers
                U_snap = np.round(U_sol * 2) / 2
                V_snap = np.round(V_sol * 2) / 2
                W_snap = np.round(W_sol * 2) / 2
                
                # Verify
                T_snap = np.zeros((n, m, p))
                for r_idx in range(r):
                    u = U_snap[:, r_idx].reshape(n, m)
                    v = V_snap[:, r_idx].reshape(m, p)
                    w = W_snap[:, r_idx].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                T_snap[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                                
                snap_error = np.linalg.norm(T - T_snap)
                
                if snap_error < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
