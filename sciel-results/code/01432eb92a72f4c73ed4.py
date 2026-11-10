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
    
    def compute_union_area_at_c(x_flat, y_flat, c):
        """Compute exact union area of 64 axis-aligned squares at height c."""
        x_starts = x_flat + c * i_indices / N
        y_starts = y_flat + c * j_indices / N
        x_ends = x_starts + delta
        y_ends = y_starts + delta
        
        # Collect unique x and y coordinates
        xs = np.unique(np.concatenate([x_starts, x_ends]))
        ys = np.unique(np.concatenate([y_starts, y_ends]))
        
        if len(xs) < 2 or len(ys) < 2:
            return 0.0
            
        total_area = 0.0
        # Iterate over x-intervals
        for xi in range(len(xs) - 1):
            x_left = xs[xi]
            x_right = xs[xi + 1]
            if x_right - x_left < 1e-12:
                continue
            
            # Find rectangles covering this x-interval
            # A rect covers [x_left, x_right] if x_start <= x_left and x_end >= x_right
            mask = (x_starts <= x_left + 1e-12) & (x_ends >= x_right - 1e-12)
            
            if not np.any(mask):
                continue
            
            # For covered rectangles, compute union of y-intervals
            y_s = y_starts[mask]
            y_e = y_ends[mask]
            
            # Sort events
            events = np.empty((2 * len(y_s), 2), dtype=np.float64)
            events[0::2, 0] = y_s
            events[0::2, 1] = 1
            events[1::2, 0] = y_e
            events[1::2, 1] = -1
            
            # Sort by y, then by type (end before start? doesn't matter for area if we handle counts correctly)
            # Actually, sorting by value is sufficient.
            order = np.argsort(events[:, 0], kind='stable')
            events = events[order]
            
            current_count = 0
            prev_y = events[0, 0]
            area_y = 0.0
            
            for y_val, etype in events:
                if current_count > 0:
                    area_y += y_val - prev_y
                current_count += etype
                prev_y = y_val
            
            total_area += (x_right - x_left) * area_y
        
        return total_area

    def estimate_volume(params):
        if time.time() - start_time > timeout:
            return np.inf
            
        x_flat = params[:64]
        y_flat = params[64:]
        
        # The area A(c) is piecewise linear in c.
        # The breakpoints occur when any two boundaries coincide.
        # x_start_k(c) = x_k + c * i_k/N
        # x_end_k(c) = x_k + delta + c * i_k/N
        # y_start_k(c) = y_k + c * j_k/N
        # y_end_k(c) = y_k + delta + c * j_k/N
        
        # We need to find c in [0,1] where any pair of these lines intersect.
        # Line m: val_m(c) = a_m + b_m * c
        # Intersection of m and n: (a_m - a_n) / (b_n - b_m)
        
        # Collect all lines
        # 4 lines per tube: x_start, x_end, y_start, y_end
        # Total 256 lines.
        # Pairs: 256*255/2 ~ 32k pairs.
        
        a_vals = np.empty(4 * 64)
        b_vals = np.empty(4 * 64)
        
        # x_start
        a_vals[0:64] = x_flat
        b_vals[0:64] = i_indices / N
        # x_end
        a_vals[64:128] = x_flat + delta
        b_vals[64:128] = i_indices / N
        # y_start
        a_vals[128:192] = y_flat
        b_vals[128:192] = j_indices / N
        # y_end
        a_vals[192:256] = y_flat + delta
        b_vals[192:256] = j_indices / N
        
        breakpoints = set([0.0, 1.0])
        
        # Vectorized intersection calculation
        # We can do this in chunks to save memory if needed, but 256 is small.
        # A matrix A[i,j] = a_i - a_j
        # A matrix B[i,j] = b_i - b_j
        # c = A[i,j] / B[i,j] if B[i,j] != 0
        
        # To avoid O(N^2) memory for 256x256 (which is fine, 65k floats), 
        # we can just loop or use broadcasting.
        
        # Broadcasting:
        # a_diff = a_vals[:, None] - a_vals[None, :]
        # b_diff = b_vals[:, None] - b_vals[None, :]
        
        # This creates 256x256 arrays. 65k elements. Very fast.
        
        a_diff = a_vals[:, None] - a_vals[None, :]
        b_diff = b_vals[:, None] - b_vals[None, :]
        
        # Mask where b_diff is not zero
        mask = np.abs(b_diff) > 1e-12
        # Calculate c
        c_vals = np.zeros_like(a_diff)
        c_vals[mask] = a_diff[mask] / b_diff[mask]
        
        # Filter c_vals to [0,1]
        valid_c = c_vals[(c_vals >= 0.0) & (c_vals <= 1.0)]
        
        # Add to breakpoints
        breakpoints.update(valid_c.tolist())
        
        # Sort breakpoints
        sorted_bp = sorted(list(breakpoints))
        
        total_volume = 0.0
        
        # Integrate A(c) over intervals defined by sorted_bp
        # A(c) is linear in each interval.
        # We can evaluate A at the endpoints of each interval and use trapezoidal rule.
        # Since it's linear, trapezoidal rule is exact.
        
        for idx in range(len(sorted_bp) - 1):
            c1 = sorted_bp[idx]
            c2 = sorted_bp[idx + 1]
            
            if c2 - c1 < 1e-12:
                continue
                
            # Evaluate area at c1 and c2
            # Note: A(c) is continuous, so we can just use midpoints or endpoints.
            # Using endpoints is exact for linear functions.
            a1 = compute_union_area_at_c(x_flat, y_flat, c1)
            a2 = compute_union_area_at_c(x_flat, y_flat, c2)
            
            total_volume += 0.5 * (a1 + a2) * (c2 - c1)
            
        return total_volume

    # Optimization
    # Use L-BFGS-B with multiple restarts from different initial points
    # Bounds: Keep positions in [-1, 1] to avoid huge bounding boxes
    bounds = [(-1.0, 1.0)] * 128
    
    best_params = initial_params.copy()
    best_volume = estimate_volume(initial_params)
    
    # Try multiple local optimizations from different starting points
    rng_opt = np.random.default_rng(42)
    num_restarts = 8 # Increased restarts since objective is faster
    
    for restart in range(num_restarts):
        if time.time() - start_time > timeout - 30:  # Leave 30s margin
            break
        
        if restart == 0:
            current_params = initial_params.copy()
        else:
            # Perturb from best found so far
            noise_scale = 0.2 / (restart + 1)
            current_params = best_params + rng_opt.normal(0, noise_scale, 128)
            current_params = np.clip(current_params, -1.0, 1.0)
        
        try:
            result = minimize(
                estimate_volume, 
                current_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'disp': False, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
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
