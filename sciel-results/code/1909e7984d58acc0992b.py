# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time


def kakeya_needle_3d8() -> np.ndarray:
    """
    Optimizes the base positions of 64 sheared tubes in 3D to minimize the union volume.
    Uses the separability of the x and y dimensions to decompose the problem into
    two 1D union length minimization problems, solved jointly via L-BFGS-B.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    N = 8
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    x_init = keich.copy()
    y_init = keich.copy()
    params_init = np.concatenate([x_init, y_init])
    
    def union_length(intervals):
        """
        Compute the total length of the union of 1D intervals.
        intervals: list of (start, end)
        """
        if not intervals:
            return 0.0
        # Sort by start
        intervals = sorted(intervals, key=lambda x: x[0])
        total_len = 0.0
        current_start, current_end = intervals[0]
        
        for s, e in intervals[1:]:
            if s > current_end:
                total_len += current_end - current_start
                current_start, current_end = s, e
            else:
                current_end = max(current_end, e)
        
        total_len += current_end - current_start
        return total_len

    def volume_estimate(params):
        """
        Compute the volume V = integral_0^1 Lx(c) Ly(c) dc
        """
        x = params[:N]
        y = params[N:]
        
        # Sample c values
        # 200 points for better accuracy
        c_vals = np.linspace(0, 1, 200)
        
        lx_arr = np.zeros_like(c_vals)
        ly_arr = np.zeros_like(c_vals)
        
        for k, c in enumerate(c_vals):
            # X intervals
            x_intervals = []
            for i in range(N):
                shift = c * i / N
                start = x[i] + shift
                end = x[i] + 1.0/N + shift
                x_intervals.append((start, end))
            lx_arr[k] = union_length(x_intervals)
            
            # Y intervals
            y_intervals = []
            for j in range(N):
                shift = c * j / N
                start = y[j] + shift
                end = y[j] + 1.0/N + shift
                y_intervals.append((start, end))
            ly_arr[k] = union_length(y_intervals)
        
        # Integrate product
        v = np.trapz(lx_arr * ly_arr, c_vals)
        return v

    # Optimization
    final_params = params_init
    try:
        # Use L-BFGS-B for constrained/efficient optimization
        result = opt.minimize(
            volume_estimate, 
            params_init, 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
        )
        
        if time.time() > deadline:
            # If we are close to deadline, stick with initial if it's better? 
            # Or just use what we have if it's valid.
            # L-BFGS-B returns the best point found so far if interrupted? 
            # Actually, if we catch a timeout, we should probably just return initial 
            # or the current best if we could access it. 
            # Since we can't easily interrupt minimize, we rely on it being fast.
            pass
        else:
            final_params = result.x
            
    except Exception:
        # If optimization fails, return initial
        final_params = params_init

    x_opt = final_params[:N]
    y_opt = final_params[N:]
    
    # Construct the 64x2 array
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Validity check
    if not np.all(np.isfinite(positions)):
        base_x = x_init[i_flat]
        base_y = y_init[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
        
    return positions

# EVOLVE-BLOCK-END
