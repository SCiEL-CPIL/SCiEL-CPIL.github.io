import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Initial guess: Symmetric construction
    # x_k = -i/(2*N), y_k = -j/(2*N)
    # This centers the spread at c=0.5
    x0 = np.zeros((N, N))
    y0 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            x0[i, j] = -i / (2.0 * N)
            y0[i, j] = -j / (2.0 * N)
            
    initial_params = np.concatenate([x0.flatten(), y0.flatten()])
    
    # Time guard
    start_time = time.time()
    timeout = 550.0 # Leave 50s margin
    
    # Pre-compute indices
    i_indices = np.repeat(np.arange(N), N)
    j_indices = np.tile(np.arange(N), N)
    
    # Exact sweep-line union area for a set of axis-aligned rectangles
    def sweep_line_union_area(x_starts, y_starts, x_ends, y_ends):
        """Compute exact union area of axis-aligned rectangles using sweep line."""
        n = len(x_starts)
        if n == 0:
            return 0.0
        
        # Collect all x and y coordinates
        xs = np.concatenate([x_starts, x_ends])
        ys = np.concatenate([y_starts, y_ends])
        
        # Sort and deduplicate
        xs = np.unique(xs)
        ys = np.unique(ys)
        
        if len(xs) < 2 or len(ys) < 2:
            return 0.0
        
        # Create grid of cells
        # For each cell, check if it's covered by at least one rectangle
        # Use coordinate compression
        cell_x_widths = np.diff(xs)
        cell_y_heights = np.diff(ys)
        
        # For each cell, determine coverage
        # A cell [xs[i], xs[i+1]] x [ys[j], ys[j+1]] is covered if
        # any rectangle has x_start <= xs[i] and x_end >= xs[i+1]
        # and y_start <= ys[j] and y_end >= ys[j+1]
        
        # Vectorized approach: for each rectangle, mark cells it covers
        # This is O(n * len(xs) * len(ys)) which could be slow
        # Better: for each x-interval, find which y-intervals are covered
        
        total_area = 0.0
        for xi in range(len(xs) - 1):
            x_left = xs[xi]
            x_right = xs[xi + 1]
            if x_right - x_left < 1e-15:
                continue
            
            # Find rectangles that span this x-interval
            # rect covers [x_left, x_right] if x_start <= x_left and x_end >= x_right
            mask = (x_starts <= x_left + 1e-12) & (x_ends >= x_right - 1e-12)
            
            if not np.any(mask):
                continue
            
            # For covered rectangles, find union of y-intervals
            y_s = y_starts[mask]
            y_e = y_ends[mask]
            
            # Sweep line in y
            events = []
            for idx in range(len(y_s)):
                events.append((y_s[idx], 1))  # start
                events.append((y_e[idx], -1))  # end
            
            events.sort()
            
            current_count = 0
            prev_y = events[0][0]
            interval_y_covered = 0.0
            
            for y_val, event_type in events:
                if current_count > 0:
                    interval_y_covered += y_val - prev_y
                current_count += event_type
                prev_y = y_val
            
            total_area += (x_right - x_left) * interval_y_covered
        
        return total_area
    
    def estimate_volume(params):
        if time.time() - start_time > timeout:
            return np.inf
            
        x_flat = params[:64]
        y_flat = params[64:]
        
        # Sample c values - use more samples for better accuracy
        num_c_samples = 30
        c_samples = np.linspace(0, 1, num_c_samples)
        
        total_volume = 0.0
        
        for c in c_samples:
            # Calculate corners
            x_starts = x_flat + c * i_indices / N
            y_starts = y_flat + c * j_indices / N
            x_ends = x_starts + delta
            y_ends = y_starts + delta
            
            area = sweep_line_union_area(x_starts, y_starts, x_ends, y_ends)
            total_volume += area
        
        # Integrate over c using trapezoidal rule
        volume = total_volume / num_c_samples
        return volume

    # Optimization
    # Use L-BFGS-B with multiple restarts from different initial points
    # Bounds: Keep positions in [-1, 1] to avoid huge bounding boxes
    bounds = [(-1.0, 1.0)] * 128
    
    best_params = initial_params.copy()
    best_volume = estimate_volume(initial_params)
    
    # Try multiple local optimizations from different starting points
    rng_opt = np.random.default_rng(42)
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 30:  # Leave 30s margin
            break
        
        if restart == 0:
            current_params = initial_params.copy()
        else:
            # Perturb from best found so far
            noise_scale = 0.1 / (restart + 1)
            current_params = best_params + rng_opt.normal(0, noise_scale, 128)
            current_params = np.clip(current_params, -1.0, 1.0)
        
        try:
            result = minimize(
                estimate_volume, 
                current_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'disp': False, 'maxls': 30, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_volume:
                best_volume = result.fun
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    optimal_params = best_params
        
    # Reconstruct positions
    x_opt = optimal_params[:64].reshape(N, N)
    y_opt = optimal_params[64:].reshape(N, N)
    
    positions = np.empty((64, 2), dtype=np.float64)
    for k in range(64):
        i = k // N
        j = k % N
        positions[k, 0] = x_opt[i, j]
        positions[k, 1] = y_opt[i, j]
        
    return positions
