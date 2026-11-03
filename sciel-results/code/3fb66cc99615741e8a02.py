import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        # T[i, j, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
        # We want T[i, j, k] == 1 for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(x):
        # Snap to nearest multiple of 0.5
        return np.round(x * 2) / 2

    def try_rank(R, max_restarts=3):
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                return None
            
            np.random.seed(42 + restart)
            
            # Initialize
            # Use schoolbook pattern for first min(R, 36) columns to guide the optimizer
            init_U = np.zeros((n*m, R))
            init_V = np.zeros((m*p, R))
            init_W = np.zeros((p*n, R))
            
            num_schoolbook = min(R, n * m * p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < num_schoolbook:
                            init_U[i*m + j, col] = 1.0
                            init_V[j*p + k, col] = 1.0
                            init_W[k*n + i, col] = 1.0
                            col += 1
            
            # Add small random noise to break symmetry and allow movement
            noise_scale = 0.1
            init_U += np.random.uniform(-noise_scale, noise_scale, init_U.shape)
            init_V += np.random.uniform(-noise_scale, noise_scale, init_V.shape)
            init_W += np.random.uniform(-noise_scale, noise_scale, init_W.shape)
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            def residuals(params):
                size_U = n * m * R
                size_V = m * p * R
                size_W = p * n * R
                
                U = params[:size_U].reshape((n*m, R))
                V = params[size_U:size_U + size_V].reshape((m*p, R))
                W = params[size_U + size_V:].reshape((p*n, R))
                
                res = np.zeros(n*m*p)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                            res[idx] = val - 1.0
                            idx += 1
                return res

            try:
                result = least_squares(
                    residuals,
                    params,
                    method='lm', # Levenberg-Marquardt, good for small problems
                    max_nfev=1000
                )
                
                if result.cost < 1e-12:
                    # Extract and snap
                    size_U = n * m * R
                    size_V = m * p * R
                    U = result.x[:size_U].reshape((n*m, R))
                    V = result.x[size_U:size_U + size_V].reshape((m*p, R))
                    W = result.x[size_U + size_V:].reshape((p*n, R))
                    
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    
                    if verify(U_snap, V_snap, W_snap, tol=1e-6):
                        return U_snap, V_snap, W_snap
                        
            except Exception:
                pass
                
        return None

    # Try ranks from 35 down to 30
    for R in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
