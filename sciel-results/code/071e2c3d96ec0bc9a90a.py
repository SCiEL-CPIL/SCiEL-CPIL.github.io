import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 630.0

    # Keich binary construction for initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X0 = keich.copy()
    Y0 = keich.copy()

    # Precompute indices for vectorization
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    def volume_estimate(X, Y, n_samples=50000, seed=12345):
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        x_min = np.min(X)
        x_max = np.max(X + (np.arange(N) + 1) / N)
        y_min = np.min(Y)
        y_max = np.max(Y + (np.arange(N) + 1) / N)
        
        box_vol = (x_max - x_min) * (y_max - y_min) * 1.0
        if box_vol <= 1e-9:
            return 1e9
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(x_min, x_max, n_samples)
        b = rng.uniform(y_min, y_max, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        a_b = a[:, np.newaxis]
        b_b = b[:, np.newaxis]
        c_b = c[:, np.newaxis]
        
        x_low = X[i_flat][np.newaxis, :] + c_b * i_flat[np.newaxis, :] / N
        x_high = x_low + 1.0 / N
        y_low = Y[j_flat][np.newaxis, :] + c_b * j_flat[np.newaxis, :] / N
        y_high = y_low + 1.0 / N
        
        in_x = (a_b >= x_low) & (a_b <= x_high)
        in_y = (b_b >= y_low) & (b_b <= y_high)
        
        in_tube = in_x & in_y
        is_inside = np.any(in_tube, axis=1)
        
        frac = np.mean(is_inside)
        return frac * box_vol

    def objective(params):
        if time.time() > deadline - 10:
            return 1e9
        X = params[:N]
        Y = params[N:]
        return volume_estimate(X, Y, n_samples=20000, seed=12345)
    
    initial_params = np.concatenate([X0, Y0])
    
    best_params = initial_params.copy()
    best_score = objective(initial_params)
    
    # Local search with Nelder-Mead
    for restart in range(5):
        if time.time() > deadline - 60:
            break
            
        if restart == 0:
            start = initial_params.copy()
        else:
            # Add noise based on restart index for diversity
            rng_noise = np.random.default_rng(restart * 100 + 1)
            noise = rng_noise.normal(0, 0.1, size=2*N)
            start = initial_params + noise
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 30:
            break
            
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
