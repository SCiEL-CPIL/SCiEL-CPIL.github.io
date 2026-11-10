import numpy as np
import scipy.optimize as opt
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Precompute tube indices
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # Initial Keich construction for X and Y separately
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    X_init = keich.copy()
    Y_init = keich.copy()
    
    # Fast MC volume for optimization
    def mc_volume_separable(p, n_samples=50000, seed=12345):
        X = p[:8]
        Y = p[8:]
        
        # Construct full positions
        # x_k = X[i], y_k = Y[j]
        x_full = X[i_flat]
        y_full = Y[j_flat]
        
        X0 = np.min(x_full)
        X1 = np.max(x_full + (i_flat + 1) / 8.0)
        Y0 = np.min(y_full)
        Y1 = np.max(y_full + (j_flat + 1) / 8.0)
        
        box_area = (X1 - X0) * (Y1 - Y0)
        if box_area <= 1e-9:
            return 1e6
        
        rng = np.random.default_rng(seed)
        a = rng.uniform(X0, X1, n_samples)
        b = rng.uniform(Y0, Y1, n_samples)
        c = rng.uniform(0, 1, n_samples)
        
        # Vectorized check
        # x_k + c*i/8
        xs = x_full[None, :] + c[:, None] * i_flat[None, :] / 8.0
        xe = xs + 1.0 / 8.0
        ys = y_full[None, :] + c[:, None] * j_flat[None, :] / 8.0
        ye = ys + 1.0 / 8.0
        
        in_x = (a[:, None] >= xs) & (a[:, None] <= xe)
        in_y = (b[:, None] >= ys) & (b[:, None] <= ye)
        in_tube = in_x & in_y
        inside = in_tube.any(axis=1)
        
        V = inside.mean() * box_area
        return V

    p0 = np.concatenate([X_init, Y_init])
    
    # Optimize the 16 separable parameters
    best_p = p0.copy()
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 60:
        try:
            # Nelder-Mead is robust for small dimensions
            res = opt.minimize(
                mc_volume_separable, best_p, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < 1e6: # Valid volume
                best_p = res.x.copy()
        except Exception:
            pass

    # Polish with L-BFGS-B if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        try:
            res2 = opt.minimize(
                mc_volume_separable, best_p, method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            if res2.fun < 1e6:
                best_p = res2.x.copy()
        except Exception:
            pass

    # Convert back to 64x2 array
    X_opt = best_p[:8]
    Y_opt = best_p[8:]
    
    x_final = X_opt[i_flat]
    y_final = Y_opt[j_flat]
    
    return np.stack([x_final, y_final], axis=1)
