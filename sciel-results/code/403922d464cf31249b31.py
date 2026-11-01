import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    num_u = n * m  # 6
    num_v = m * p  # 21
    num_w = p * n  # 14
    
    # Target ranks to try
    target_ranks = [41, 40, 39, 38, 37, 36]
    
    # Precompute target tensor for fast error computation
    # Target: Z[i,j,k,a,b] = 1 if i==a and k==b else 0
    # Shape: (n, m, p, n, p)
    target = np.zeros((n, m, p, n, p))
    for i in range(n):
        for k in range(p):
            target[i, :, k, i, k] = 1.0
    
    target_flat = target.flatten()
    
    # Precompute indices for reshaping
    # U: (R, num_u) -> reshape to (n, m, R)
    # V: (R, num_v) -> reshape to (m, p, R)
    # W: (R, num_w) -> reshape to (p, n, R)
    
    def error_func(params, R):
        # params is flat array of size R * (num_u + num_v + num_w)
        U_flat = params[:R*num_u]
        V_flat = params[R*num_u:R*(num_u+num_v)]
        W_flat = params[R*(num_u+num_v):]
        
        # Reshape to 3D tensors
        # U: (n, m, R) - note: U[i*m+j, r] -> U[i, j, r]
        U = U_flat.reshape(n, m, R)
        # V: (m, p, R) - V[j*p+k, r] -> V[j, k, r]
        V = V_flat.reshape(m, p, R)
        # W: (p, n, R) - W[k*n+i, r] -> W[k, i, r]
        W = W_flat.reshape(p, n, R)
        
        # Compute Z[i,j,k,a,b] = sum_r U[i,j,r] * V[j,k,r] * W[b,a,r]
        # Using einsum: 'ijr,jkr,bar->ijkab'
        Z = np.einsum('ijr,jkr,bar->ijkab', U, V, W, optimize=True)
        
        return (Z - target).flatten()
    
    best_result = None
    best_rank = n * m * p  # 42
    
    found_solution = False
    for R in target_ranks:
        if time.time() > deadline:
            break
            
        total_vars = R * (num_u + num_v + num_w)
        
        # Try multiple random initializations
        num_restarts = 10
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + restart * 1000 + R)
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                from scipy.optimize import least_squares
                res = least_squares(error_func, x0, args=(R,), method='lm', max_nfev=2000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            except Exception:
                continue
                
            if res.cost < 1e-10:
                # Check if we can snap to half-integers
                U_flat = res.x[:R*num_u]
                V_flat = res.x[R*num_u:R*(num_u+num_v)]
                W_flat = res.x[R*(num_u+num_v):]
                
                U = U_flat.reshape(n, m, R)
                V = V_flat.reshape(m, p, R)
                W = W_flat.reshape(p, n, R)
                
                # Snap to nearest half-integer
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                if np.allclose(U_snap, U, atol=1e-8) and np.allclose(V_snap, V, atol=1e-8) and np.allclose(W_snap, W, atol=1e-8):
                    # Re-verify with snapped values
                    params_snap = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                    err = error_func(params_snap, R)
                    if np.max(np.abs(err)) < 1e-8:
                        # Success - convert back to 2D format
                        U_2d = U_snap.transpose(2, 0, 1).reshape(R, num_u).T
                        V_2d = V_snap.transpose(2, 0, 1).reshape(R, num_v).T
                        W_2d = W_snap.transpose(2, 0, 1).reshape(R, num_w).T
                        
                        return {
                            "U": U_2d.tolist(),
                            "V": V_2d.tolist(),
                            "W": W_2d.tolist()
                        }
        
    # Fallback to schoolbook rank 42
    rank = n * m * p
    U = [[0] * rank for _ in range(num_u)]
    V = [[0] * rank for _ in range(num_v)]
    W = [[0] * rank for _ in range(num_w)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
