import numpy as np
import time

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    w = 1.0 / N  # tube width = 0.125
    start_time = time.time()
    deadline = 630.0  # wall-clock limit; keep a safety margin
    
    # Precompute grids
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()  # shape (64,)
    j_flat = j_grid.flatten()  # shape (64,)
    i_frac = i_flat / N  # (64,)
    j_frac = j_flat / N  # (64,)
    
    def compute_volume_fast(xk, yk, num_samples=50000):
        """Fast volume estimator for local search. xk, yk are shape (64,)."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_frac)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_frac)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        # Vectorized check: for each sample, check if in any tube
        c_col = c[:, np.newaxis]  # (M, 1)
        x_left = xk[np.newaxis, :] + c_col * i_frac[np.newaxis, :]  # (M, 64)
        y_left = yk[np.newaxis, :] + c_col * j_frac[np.newaxis, :]  # (M, 64)
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    def compute_volume_full(xk, yk, num_samples=500000):
        """Full precision volume estimator for final evaluation."""
        X0 = np.min(xk)
        X1 = np.max(xk + i_frac)
        Y0 = np.min(yk)
        Y1 = np.max(yk + j_frac)
        
        area = (X1 - X0) * (Y1 - Y0)
        if area <= 1e-9:
            return float('inf')
        
        rng = np.random.default_rng(12345)
        a = rng.uniform(X0, X1, num_samples)
        b = rng.uniform(Y0, Y1, num_samples)
        c = rng.uniform(0, 1, num_samples)
        
        c_col = c[:, np.newaxis]
        x_left = xk[np.newaxis, :] + c_col * i_frac[np.newaxis, :]
        y_left = yk[np.newaxis, :] + c_col * j_frac[np.newaxis, :]
        
        in_x = (a[:, np.newaxis] >= x_left) & (a[:, np.newaxis] <= x_left + w)
        in_y = (b[:, np.newaxis] >= y_left) & (b[:, np.newaxis] <= y_left + w)
        in_any = np.any(in_x & in_y, axis=1)
        
        fraction = np.mean(in_any)
        volume = fraction * area
        return volume
    
    # Initial configuration: symmetric "fan" construction
    # x_k depends only on i, y_k depends only on j
    # x_i = -i * alpha, y_j = -j * alpha
    # This makes tubes converge toward the center as c increases
    
    def initial_config(alpha=0.0625):
        xk = -i_flat * alpha
        yk = -j_flat * alpha
        return xk, yk
    
    # Start with a reasonable initial guess
    xk, yk = initial_config(0.0625)
    best_xk, best_yk = xk.copy(), yk.copy()
    best_vol = compute_volume_fast(xk, yk, 50000)
    
    # Coordinate descent local search with adaptive step size
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 150
    min_step = 1e-7
    
    while time.time() - start_time < deadline - 30:  # Leave 30s margin
        improved = False
        for k in range(64):
            if time.time() - start_time > deadline - 30:
                break
            
            # Try small perturbations in x and y
            best_local_vol = best_vol
            best_local_xk = xk.copy()
            best_local_yk = yk.copy()
            
            for dx in [-step_size, step_size]:
                for dy in [-step_size, step_size]:
                    xk_new = xk.copy()
                    yk_new = yk.copy()
                    xk_new[k] += dx
                    yk_new[k] += dy
                    
                    vol = compute_volume_fast(xk_new, yk_new, 50000)
                    if vol < best_local_vol - 1e-10:
                        best_local_vol = vol
                        best_local_xk = xk_new
                        best_local_yk = yk_new
                        improved = True
            
            if improved:
                xk, yk = best_local_xk, best_local_yk
                best_vol = best_local_vol
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            no_improve_count = 0
            if step_size < min_step:
                break
    
    # Keep track of the best configuration found
    best_xk = xk.copy()
    best_yk = yk.copy()
    
    # Final evaluation with full samples
    final_vol = compute_volume_full(best_xk, best_yk, 500000)
    
    # Reshape to (64, 2)
    positions = np.stack([best_xk, best_yk], axis=1)
    return positions
