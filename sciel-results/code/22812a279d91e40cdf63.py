import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 600  # Safety margin
    
    N = 8
    n_samples = 500000
    
    # Pre-compute indices for tubes
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    def compute_volume(params):
        """
        Compute Monte Carlo volume estimate for given parameters.
        params: array of shape (32,) containing A, B, C, D
        """
        if time.time() > deadline:
            return 1e10
            
        A = params[0:N]
        B = params[N:2*N]
        C = params[2*N:3*N]
        D = params[3*N:4*N]
        
        # Construct base positions
        # x_ij = A_i + B_j
        # y_ij = C_i + D_j
        x_coords = A[:, None] + B[None, :]
        y_coords = C[:, None] + D[None, :]
        
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        
        # Determine bounds
        X0 = np.min(x_flat)
        X1 = np.max(x_flat + (i_flat + 1) / N)
        Y0 = np.min(y_flat)
        Y1 = np.max(y_flat + (j_flat + 1) / N)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return 1e10
            
        # Generate points
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Check if points are inside any tube
        inside = np.zeros(n_samples, dtype=bool)
        
        for k in range(N * N):
            i_k = i_flat[k]
            j_k = j_flat[k]
            x_k = x_flat[k]
            y_k = y_flat[k]
            
            a_min = x_k + c * i_k / N
            a_max = a_min + 1.0 / N
            b_min = y_k + c * j_k / N
            b_max = b_min + 1.0 / N
            
            mask = (a >= a_min) & (a <= a_max) & (b >= b_min) & (b <= b_max)
            inside |= mask
            
        fraction = np.mean(inside)
        volume = fraction * area
        return volume

    # Initial guess
    # A_i = -i/8, B_j = 0, C_i = 0, D_j = -j/8
    A_init = -np.arange(N) / N
    B_init = np.zeros(N)
    C_init = np.zeros(N)
    D_init = -np.arange(N) / N
    
    params_init = np.concatenate([A_init, B_init, C_init, D_init])
    
    best_params = params_init.copy()
    best_volume = compute_volume(best_params)
    
    # Optimization
    try:
        res = minimize(compute_volume, params_init, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
        if res.fun < best_volume:
            best_params = res.x
            best_volume = res.fun
    except Exception:
        pass
    
    # If time remains, try to refine with full 128 parameters?
    # Might be too slow. Let's stick to separable for now.
    # If we have lots of time, we can do a second pass.
    
    if time.time() < deadline - 120:
        # Try a few random restarts or different initializations
        inits = [
            np.concatenate([np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)]),
            np.concatenate([np.arange(N)/N, np.zeros(N), np.zeros(N), np.arange(N)/N]),
            np.concatenate([-np.arange(N)/N, np.zeros(N), np.zeros(N), -np.arange(N)/N])
        ]
        
        for init in inits:
            if time.time() > deadline - 60:
                break
            try:
                res_r = minimize(compute_volume, init, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False})
                if res_r.fun < best_volume:
                    best_params = res_r.x
                    best_volume = res_r.fun
            except Exception:
                pass

    # Construct final positions
    A = best_params[0:N]
    B = best_params[N:2*N]
    C = best_params[2*N:3*N]
    D = best_params[3*N:4*N]
    
    x_coords = A[:, None] + B[None, :]
    y_coords = C[:, None] + D[None, :]
    
    positions = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
    
    return positions
