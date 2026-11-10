import numpy as np
import time
from scipy.optimize import minimize

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
    
    # Define the deterministic volume estimator
    # Uses a grid to approximate the union area
    def estimate_volume(params):
        if time.time() - start_time > timeout:
            return np.inf
            
        x_flat = params[:64]
        y_flat = params[64:]
        
        # Reshape to (i, j)
        x_pos = x_flat.reshape(N, N)
        y_pos = y_flat.reshape(N, N)
        
        # Sample c values
        num_c_samples = 20
        c_samples = np.linspace(0, 1, num_c_samples)
        
        total_volume = 0.0
        
        # Grid resolution
        # The tubes move within a range. 
        # Max x shift is 7/8. Min x is -7/16. Range approx 1.
        # Let's use a grid that covers [-1, 2] for safety, size 100x100
        # Or dynamically determine bounds? 
        # Dynamic bounds are better to keep grid tight.
        
        # Pre-compute indices
        i_indices = np.repeat(np.arange(N), N)
        j_indices = np.tile(np.arange(N), N)
        
        for c in c_samples:
            # Calculate corners
            # x_k + c * i / 8
            x_starts = x_flat + c * i_indices / N
            y_starts = y_flat + c * j_indices / N
            
            x_ends = x_starts + delta
            y_ends = y_starts + delta
            
            # Find bounding box
            min_x = np.min(x_starts)
            max_x = np.max(x_ends)
            min_y = np.min(y_starts)
            max_y = np.max(y_ends)
            
            width = max_x - min_x
            height = max_y - min_y
            
            if width < 1e-9 or height < 1e-9:
                continue
                
            # Grid resolution
            # Aim for ~100x100 grid
            # But keep pixel size consistent? 
            # Let's use a fixed number of bins relative to the bounding box
            num_bins_x = 100
            num_bins_y = 100
            
            # Create bin edges
            x_edges = np.linspace(min_x, max_x, num_bins_x + 1)
            y_edges = np.linspace(min_y, max_y, num_bins_y + 1)
            
            # Rasterize
            # For each rectangle, determine which bins it covers
            # This is a bit slow if done in a loop.
            # Alternative: Use a boolean array and slice assignment.
            
            grid = np.zeros((num_bins_y, num_bins_x), dtype=bool)
            
            for k in range(64):
                x_s = x_starts[k]
                x_e = x_ends[k]
                y_s = y_starts[k]
                y_e = y_ends[k]
                
                # Find bin indices
                # Clip to valid range
                bin_x_start = np.searchsorted(x_edges, x_s, side='right') - 1
                bin_x_end = np.searchsorted(x_edges, x_e, side='left')
                bin_y_start = np.searchsorted(y_edges, y_s, side='right') - 1
                bin_y_end = np.searchsorted(y_edges, y_e, side='left')
                
                # Ensure valid indices
                bin_x_start = np.clip(bin_x_start, 0, num_bins_x - 1)
                bin_x_end = np.clip(bin_x_end, 0, num_bins_x)
                bin_y_start = np.clip(bin_y_start, 0, num_bins_y - 1)
                bin_y_end = np.clip(bin_y_end, 0, num_bins_y)
                
                if bin_x_start < bin_x_end and bin_y_start < bin_y_end:
                    grid[bin_y_start:bin_y_end, bin_x_start:bin_x_end] = True
            
            area = np.sum(grid) * (width / num_bins_x) * (height / num_bins_y)
            total_volume += area
            
        # Integrate over c
        # Trapezoidal rule
        volume = total_volume / num_c_samples
        return volume

    # Optimization
    # Use L-BFGS-B
    # Bounds: Keep positions in [-1, 1] to avoid huge bounding boxes
    bounds = [(-1.0, 1.0)] * 128
    
    try:
        result = minimize(
            estimate_volume, 
            initial_params, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 100, 'disp': False, 'maxls': 20}
        )
        
        if result.success or result.fun < estimate_volume(initial_params):
            optimal_params = result.x
        else:
            optimal_params = initial_params
            
    except Exception:
        optimal_params = initial_params
        
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
