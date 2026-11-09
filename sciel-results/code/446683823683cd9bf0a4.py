import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
                
    def loss_fn(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U + size_V]
        W_flat = params[size_U + size_V:]
        
        U = U_flat.reshape(n, m, R)
        V = V_flat.reshape(m, p, R)
        W = W_flat.reshape(p, n, R)
        
        # T_calc[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # U: (n, m, R) -> indices i, j, r
        # V: (m, p, R) -> indices j, k, r
        # W: (p, n, R) -> indices k, i, r
        T_calc = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        err = T_calc - 1.0
        return np.sum(err * err)

    # Try ranks
    # 30 is the theoretical min.
    # Let's try 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89
    ranks_to_try = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 89]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
            
        for restart in range(10):
            if time.time() > deadline:
                break
                
            # Initialize
            # Random normal
            U_init = np.random.randn(n, m, R)
            V_init = np.random.randn(m, p, R)
            W_init = np.random.randn(p, n, R)
            
            # Scale to avoid large gradients initially?
            # The product is roughly R * scale^3. We want sum to be 1.
            # So scale ~ R^(-1/3).
            scale = R ** (-1/3)
            U_init *= scale
            V_init *= scale
            W_init *= scale
            
            params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                from scipy.optimize import minimize
                res = minimize(loss_fn, params, args=(R,), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14})
                
                if res.fun < 1e-6:
                    # Snap
                    U_opt = res.x[:n*m*R].reshape(n, m, R)
                    V_opt = res.x[n*m*R:n*m*R+m*p*R].reshape(m, p, R)
                    W_opt = res.x[n*m*R+m*p*R:].reshape(p, n, R)
                    
                    U_snap = np.round(U_opt * 2) / 2
                    V_snap = np.round(V_opt * 2) / 2
                    W_snap = np.round(W_opt * 2) / 2
                    
                    # Verify
                    T_snap = np.einsum('ijr,jkr,kir->ijk', U_snap, V_snap, W_snap)
                    if np.allclose(T_snap, 1.0, atol=1e-5):
                        U_list = U_snap.reshape(n*m, R).tolist()
                        V_list = V_snap.reshape(m*p, R).tolist()
                        W_list = W_snap.reshape(p*n, R).tolist()
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass

    return {"U": U_fb, "V": V_fb, "W": W_fb}
