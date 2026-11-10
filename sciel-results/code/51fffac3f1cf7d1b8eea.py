import numpy as np
import time
import sys
from scipy.optimize import minimize

def compute_volume_fast(positions):
    """
    Computes the exact volume of the union of 64 sheared tubes.
    positions: (64, 2) array.
    Returns volume.
    """
    N = 8
    base_x = positions[:, 0]
    base_y = positions[:, 1]
    
    i_idx = np.arange(64) // N
    j_idx = np.arange(64) % N
    
    # Precompute slopes
    sx = i_idx / 8.0
    sy = j_idx / 8.0
    
    # Breakpoints in c where relative ordering of x or y edges changes.
    # Edges for tube k: x_k(c) = bx_k + c*sx_k, y_k(c) = by_k + c*sy_k.
    # We need breakpoints where x_k(c) = x_m(c) +/- 1/8 or y_k(c) = y_m(c) +/- 1/8.
    
    breakpoints = np.array([0.0, 1.0])
    
    # Vectorized breakpoint generation is tricky due to conditionals, 
    # but 64 choose 2 is 2016 pairs. We can do it in a loop for simplicity 
    # or vectorize with masks. Given 600s budget, even a python loop of 2016 
    # per volume eval is fine if volume eval is fast.
    
    for k in range(64):
        for m in range(k + 1, 64):
            # X-dimension
            dx = base_x[m] - base_x[k]
            dsx = sx[m] - sx[k]
            if abs(dsx) > 1e-12:
                # x_m(c) - x_k(c) = 0 => c = -dx/dsx
                c1 = -dx / dsx
                # x_m(c) - x_k(c) = 1/8 => c = (1/8 - dx)/dsx
                c2 = (0.125 - dx) / dsx
                # x_m(c) - x_k(c) = -1/8 => c = (-1/8 - dx)/dsx
                c3 = (-0.125 - dx) / dsx
                
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        breakpoints = np.append(breakpoints, c_val)
                        
            # Y-dimension
            dy = base_y[m] - base_y[k]
            dsy = sy[m] - sy[k]
            if abs(dsy) > 1e-12:
                c1 = -dy / dsy
                c2 = (0.125 - dy) / dsy
                c3 = (-0.125 - dy) / dsy
                
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        breakpoints = np.append(breakpoints, c_val)
                        
    breakpoints = np.unique(breakpoints)
    
    total_vol = 0.0
    
    # For each interval, compute area at midpoint
    # To speed up, we can batch process if we had many, but here we just loop.
    for i in range(len(breakpoints) - 1):
        c0 = breakpoints[i]
        c1 = breakpoints[i+1]
        c_mid = 0.5 * (c0 + c1)
        dc = c1 - c0
        
        # Compute x and y ranges for all 64 tubes at c_mid
        x_starts = base_x + c_mid * sx
        y_starts = base_y + c_mid * sy
        x_ends = x_starts + 0.125
        y_ends = y_starts + 0.125
        
        # Sweep line to compute union area of 64 rectangles
        # Rectangles are (x_start, x_end, y_start, y_end)
        
        # Create events: (x_coord, delta, y_start, y_end)
        # delta = 1 for start, -1 for end
        events = np.empty((128, 4))
        events[:64, 0] = x_starts
        events[:64, 1] = 1
        events[:64, 2] = y_starts
        events[:64, 3] = y_ends
        
        events[64:, 0] = x_ends
        events[64:, 1] = -1
        events[64:, 2] = y_starts
        events[64:, 3] = y_ends
        
        # Sort by x
        sort_idx = np.argsort(events[:, 0])
        events = events[sort_idx]
        
        # We need to maintain a set of active y-intervals and compute their union length.
        # Since N=64 is small, we can just keep a list of active y-intervals and recompute union length at each x-change.
        # Number of x-changes is at most 128.
        
        active_intervals = []
        current_x = events[0, 0]
        area = 0.0
        
        for i_evt in range(128):
            x_evt = events[i_evt, 0]
            dx = x_evt - current_x
            
            if dx > 1e-12:
                # Compute union length of active_intervals
                if active_intervals:
                    # Sort by y_start
                    active_intervals.sort(key=lambda x: x[0])
                    merged = []
                    cur_s, cur_e = active_intervals[0]
                    for s, e in active_intervals[1:]:
                        if s <= cur_e:
                            if e > cur_e:
                                cur_e = e
                        else:
                            merged.append((cur_s, cur_e))
                            cur_s, cur_e = s, e
                    merged.append((cur_s, cur_e))
                    y_len = sum(e - s for s, e in merged)
                else:
                    y_len = 0.0
                area += y_len * dx
                
            current_x = x_evt
            
            # Update active intervals
            delta = events[i_evt, 1]
            y_s = events[i_evt, 2]
            y_e = events[i_evt, 3]
            
            if delta == 1:
                active_intervals.append((y_s, y_e))
            else:
                # Remove matching interval
                # Since floating point equality can be tricky, we rely on the fact 
                # that we added exactly this interval. 
                # To be safe, we can use a multiset or just pop the first match.
                for idx, (s, e) in enumerate(active_intervals):
                    if np.isclose(s, y_s) and np.isclose(e, y_e):
                        active_intervals.pop(idx)
                        break
                        
        total_vol += area * dc
        
    return total_vol

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = 600.0  # Safety margin
    
    N = 8
    num_tubes = 64
    
    # Initial guess: Keich-inspired 1D positions mapped to 2D
    # Or random
    rng = np.random.default_rng(12345)
    
    # Let's try a few different initializations and pick the best
    # 1. Keich 1D -> 2D
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    init_1 = np.stack([keich[i_flat], keich[j_flat]], axis=1)
    
    # 2. Random
    init_2 = rng.uniform(-0.5, 1.5, (num_tubes, 2))
    
    # 3. Grid
    xs = rng.uniform(0, 1, 8)
    ys = rng.uniform(0, 1, 8)
    init_3 = np.stack([np.tile(xs, 8), np.repeat(ys, 8)], axis=1)
    
    # Evaluate initial volumes
    v1 = compute_volume_fast(init_1)
    v2 = compute_volume_fast(init_2)
    v3 = compute_volume_fast(init_3)
    
    best_init = init_1
    best_vol = v1
    if v2 < best_vol:
        best_init = init_2
        best_vol = v2
    if v3 < best_vol:
        best_init = init_3
        best_vol = v3
        
    best_positions = best_init.copy()
    
    # CMA-ES Optimization
    # Since we don't have py-cma, we will implement a simple CMA-ES or use scipy's differential_evolution
    # Differential Evolution is robust and available in scipy.
    # It handles the 128 dimensions (64 x 2).
    
    # Bounds:
    # x_k can be roughly in [-1, 2]
    # y_k can be roughly in [-1, 2]
    bounds = [(-1.0, 2.0)] * num_tubes + [(-1.0, 2.0)] * num_tubes
    
    # Flatten positions for scipy
    flat_init = best_init.flatten()
    
    def objective(flat_pos):
        pos = flat_pos.reshape(num_tubes, 2)
        # Check for non-finite
        if not np.all(np.isfinite(pos)):
            return 1e10
        vol = compute_volume_fast(pos)
        if vol <= 1e-9:
            return 1e10
        return vol
        
    # Run Differential Evolution
    # maxiter=100 might be too low for 128 dims, but we have time.
    # tol=1e-8
    # polish=True
    
    try:
        res = minimize(
            objective, 
            flat_init, 
            method='differential_evolution', 
            bounds=bounds,
            maxiter=200,
            tol=1e-8,
            polish=True,
            seed=42,
            workers=1 # Single process to avoid overhead/complexity
        )
        
        if res.fun < best_vol:
            best_vol = res.fun
            best_positions = res.x.reshape(num_tubes, 2)
            
    except Exception as e:
        print(f"DE failed: {e}", file=sys.stderr)
        
    # If we have time left, try a local polish with Nelder-Mead on the best found
    if time.time() - start_time < deadline - 30:
        try:
            res_nm = minimize(
                objective, 
                best_positions.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res_nm.fun < best_vol:
                best_vol = res_nm.fun
                best_positions = res_nm.x.reshape(num_tubes, 2)
        except Exception as e:
            print(f"NM failed: {e}", file=sys.stderr)
            
    # Final check
    if not np.all(np.isfinite(best_positions)):
        # Fallback to initial
        best_positions = init_1
        
    return best_positions

if __name__ == "__main__":
    res = kakeya_needle_3d8()
    print(res.shape)
