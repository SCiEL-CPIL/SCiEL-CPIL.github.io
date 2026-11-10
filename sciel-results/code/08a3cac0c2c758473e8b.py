import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0  # Safety margin
    
    # 1. Initial Guess: Keich Binary Construction
    pos_k = int(np.log2(N))
    j_terms = np.arange(1, pos_k + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos_k - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos_k)
    
    best_X = keich.copy()
    best_Y = keich.copy()
    
    # Precompute indices for the 64 tubes
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    # 2. Smooth differentiable surrogate for 1D union length
    # Use a soft-max approximation of the merge operation
    
    def _smooth_union_length(starts, widths, c_val, i_indices, alpha=100.0):
        """
        Compute a smooth approximation of the union length of intervals.
        Uses a soft-max to smoothly approximate the max operation in interval merging.
        """
        s = starts + c_val * i_indices / N
        e = s + widths
        
        # Sort by start point
        order = np.argsort(s)
        s_sorted = s[order]
        e_sorted = e[order]
        
        # Smooth merge: for each pair, the "gap" is s_{k+1} - e_k
        # If gap > 0, intervals are disjoint; if gap < 0, they overlap
        # We want to smoothly compute the total length
        
        # Alternative approach: use the fact that union length = max(e) - min(s) - sum of gaps
        # where gaps are positive parts of (s_{k+1} - e_k)
        
        # Compute all pairwise differences for adjacent intervals after sorting
        # For N intervals, there are N-1 adjacent pairs
        
        # Total length = e_sorted[-1] - s_sorted[0] - sum_{k=0}^{N-2} max(0, s_sorted[k+1] - e_sorted[k])
        # But this is only correct if intervals are properly nested after sorting
        
        # Better: use the standard merge algorithm but with soft-max for the max operation
        
        # Let's use a different approach: the union length can be computed as
        # integral over x of indicator that x is in at least one interval
        # For smooth approximation, use a soft indicator
        
        # Actually, let's use a simpler smooth approximation:
        # The union length is approximately the length of the convex hull minus the "holes"
        # But holes are hard to approximate smoothly
        
        # Let's use a grid-based soft approximation:
        # Sample many points and use a smooth max to determine coverage
        
        # For efficiency, let's use a different strategy:
        # The exact union length for sorted intervals can be computed by tracking
        # the current rightmost end. We can smooth this with a soft-max.
        
        # Simpler approach: use the formula with soft-max for the max operation
        # in the merge step
        
        # For now, let's use a numerical derivative-friendly approach:
        # Compute the union length using a smooth approximation of the indicator function
        
        # Sample points on a fine grid
        grid_size = 200
        x_min = np.min(s_sorted) - 0.1
        x_max = np.max(e_sorted) + 0.1
        grid = np.linspace(x_min, x_max, grid_size)
        
        # For each grid point, compute the soft max of (e_i - x) for all i
        # If soft_max(e_i - x) > 0, the point is covered
        # Coverage indicator = sigmoid(alpha * soft_max(e_i - x))
        
        # Compute e_i - x for all i and grid points
        # e_sorted has shape (N,), grid has shape (grid_size,)
        # We want a matrix of shape (N, grid_size)
        diff = e_sorted[:, None] - grid[None, :]
        
        # Soft max: alpha * (log(sum(exp(diff/alpha))) - log(N))
        # For numerical stability, subtract max first
        diff_scaled = diff / alpha
        diff_max = np.max(diff_scaled, axis=0, keepdims=True)
        exp_diff = np.exp(diff_scaled - diff_max)
        sum_exp = np.sum(exp_diff, axis=0)
        soft_max = alpha * (np.log(sum_exp) + diff_max[0] - np.log(N))
        
        # Coverage indicator: sigmoid(alpha * soft_max)
        # Clip for numerical stability
        soft_max_clipped = np.clip(soft_max, -50, 50)
        coverage = 1.0 / (1.0 + np.exp(-alpha * soft_max_clipped))
        
        # Union length = integral of coverage over x
        dx = (x_max - x_min) / (grid_size - 1)
        union_len = np.sum(coverage) * dx
        
        return union_len

    def compute_smooth_volume(X, Y, num_c_samples=100, alpha=100.0):
        """
        Compute the smooth union volume assuming separability.
        X: array of shape (8,) for x-base positions of rows i
        Y: array of shape (8,) for y-base positions of cols j
        """
        c_vals = np.linspace(0, 1, num_c_samples)
        
        i_idx = np.arange(N, dtype=np.float64)
        j_idx = np.arange(N, dtype=np.float64)
        
        width = 1.0 / N
        
        total_vol = 0.0
        
        for c in c_vals:
            Lx = _smooth_union_length(X, width, c, i_idx, alpha)
            Ly = _smooth_union_length(Y, width, c, j_idx, alpha)
            total_vol += Lx * Ly
        
        if num_c_samples > 1:
            weights = np.ones(num_c_samples)
            weights[0] = 0.5
            weights[-1] = 0.5
            total_vol = np.sum(total_vol * weights) * (1.0 / (num_c_samples - 1))
        
        return total_vol

    def compute_exact_volume(X, Y, num_c_samples=200):
        """
        Compute the exact union volume assuming separability.
        """
        c_vals = np.linspace(0, 1, num_c_samples)
        
        i_idx = np.arange(N, dtype=np.float64)
        j_idx = np.arange(N, dtype=np.float64)
        
        width = 1.0 / N
        
        def _union_length_1d(starts, widths, c_val, i_indices):
            s = starts + c_val * i_indices / N
            e = s + widths
            
            order = np.argsort(s)
            s_sorted = s[order]
            e_sorted = e[order]
            
            if len(s_sorted) == 0:
                return 0.0
                
            total_len = 0.0
            current_start = s_sorted[0]
            current_end = e_sorted[0]
            
            for k in range(1, len(s_sorted)):
                if s_sorted[k] > current_end:
                    total_len += current_end - current_start
                    current_start = s_sorted[k]
                    current_end = e_sorted[k]
                else:
                    if e_sorted[k] > current_end:
                        current_end = e_sorted[k]
                        
            total_len += current_end - current_start
            return total_len
        
        total_vol = 0.0
        
        for c in c_vals:
            Lx = _union_length_1d(X, width, c, i_idx)
            Ly = _union_length_1d(Y, width, c, j_idx)
            total_vol += Lx * Ly
            
        if num_c_samples > 1:
            weights = np.ones(num_c_samples)
            weights[0] = 0.5
            weights[-1] = 0.5
            total_vol = np.sum(total_vol * weights) * (1.0 / (num_c_samples - 1))
        
        return total_vol

    # 3. Optimization with L-BFGS-B using smooth surrogate
    
    def smooth_objective(params):
        if time.time() > deadline - 10:
            return 1e18
        X = params[:N]
        Y = params[N:]
        return compute_smooth_volume(X, Y, num_c_samples=80, alpha=100.0)
    
    initial_params = np.concatenate([best_X, best_Y])
    
    best_params = initial_params.copy()
    best_score = compute_exact_volume(best_params[:N], best_params[N:], num_c_samples=200)
    
    rng_restart = np.random.default_rng(12345)
    
    # Multi-restart L-BFGS-B
    num_restarts = 30
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        if restart == 0:
            start_params = initial_params.copy()
        else:
            noise_scale = 0.3
            noise = rng_restart.normal(0, noise_scale, size=2*N)
            start_params = best_params + noise
            
        try:
            res = minimize(
                smooth_objective, 
                start_params, 
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            
            # Evaluate the exact volume at the result
            exact_vol = compute_exact_volume(res.x[:N], res.x[N:], num_c_samples=200)
            
            if exact_vol < best_score:
                best_score = exact_vol
                best_params = res.x.copy()
                
        except Exception:
            pass

    # Final refinement with Nelder-Mead on exact objective
    if time.time() < deadline - 5:
        def final_objective(params):
            X = params[:N]
            Y = params[N:]
            return compute_exact_volume(X, Y, num_c_samples=300)
            
        res_final = minimize(
            final_objective,
            best_params,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        
        if res_final.fun < best_score:
            best_params = res_final.x.copy()
            best_score = res_final.fun

    # 4. Construct Output
    X_opt = best_params[:N]
    Y_opt = best_params[N:]
    
    base_x = X_opt[i_flat]
    base_y = Y_opt[j_flat]
    
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions
