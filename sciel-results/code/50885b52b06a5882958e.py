import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 20.0
    
    # Pre-compute the sofa grid
    i_arr = np.arange(450)
    j_arr = np.arange(300)
    I, J = np.meshgrid(i_arr, j_arr, indexing='ij')
    x_grid = -3 + 0.01 * I
    y_grid = -1.5 + 0.01 * J
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = x_flat.size  # 135000
    
    # Pre-compute rotations for 91 angles
    a_vals = np.arange(91)
    rad_vals = np.deg2rad(a_vals)
    cos_vals = np.cos(rad_vals)
    sin_vals = np.sin(rad_vals)
    
    def compute_score(xs, ys, verbose=False):
        """
        Compute score for a path defined by xs, ys (length 91).
        Uses chunking with early exit.
        """
        mask = np.ones(n_points, dtype=bool)
        chunk_size = 20  # Larger chunk for better vectorization
        
        for start_k in range(0, 91, chunk_size):
            end_k = min(start_k + chunk_size, 91)
            k_slice = slice(start_k, end_k)
            
            xs_chunk = xs[k_slice]  # shape (chunk_size,)
            ys_chunk = ys[k_slice]
            cos_chunk = cos_vals[k_slice]
            sin_chunk = sin_vals[k_slice]
            
            # dx shape: (chunk_size, n_points)
            dx = x_flat[None, :] - xs_chunk[:, None]
            dy = y_flat[None, :] - ys_chunk[:, None]
            
            u = dx * cos_chunk[:, None] + dy * sin_chunk[:, None]
            v = -dx * sin_chunk[:, None] + dy * cos_chunk[:, None]
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid_chunk = cond1 | cond2  # shape (chunk_size, n_points)
            
            valid_all = valid_chunk.all(axis=0)  # shape (n_points,)
            mask &= valid_all
            
            # Early exit: if no points survive, score is 0
            if not np.any(mask):
                if verbose:
                    print(f"  Early exit at k={start_k}")
                return 0.0
        
        return np.sum(mask) * 0.0001

    def compute_score_fast(xs, ys):
        """
        Optimized score computation with aggressive early exit.
        Process angles in order of expected difficulty.
        """
        # Order: start with angles likely to be most constraining
        # Angles 0, 45, 90 are the endpoints and likely most constraining
        order = [0, 90, 45, 30, 60, 15, 75, 20, 70, 25, 65, 10, 80, 35, 55, 5, 85, 40, 50, 12, 78, 22, 68, 8, 82, 28, 62, 18, 72, 33, 57, 7, 83, 27, 63, 17, 73, 38, 52, 14, 76, 24, 66, 9, 81, 32, 58, 19, 71, 37, 53, 13, 77, 23, 67, 6, 84, 31, 59, 16, 74, 36, 54, 11, 79, 26, 64, 21, 69, 4, 86, 34, 56, 29, 61, 2, 88, 39, 51, 20, 70]
        # Deduplicate and ensure all 91 are present
        order = list(set(order))
        # Add any missing
        for k in range(91):
            if k not in order:
                order.append(k)
        
        mask = np.ones(n_points, dtype=bool)
        
        for k in order:
            cx = xs[k]
            cy = ys[k]
            ca = cos_vals[k]
            sa = sin_vals[k]
            
            dx = x_flat - cx
            dy = y_flat - cy
            
            u = dx * ca + dy * sa
            v = -dx * sa + dy * ca
            
            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2
            
            mask &= valid
            
            if not np.any(mask):
                return 0.0
        
        return np.sum(mask) * 0.0001

    # Heuristics for initial paths
    def make_linear_path():
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            ys[k] = 0.0
        return xs, ys

    def make_arc_path(amplitude=1.5, x_power=1.0):
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * (t ** x_power)
            ys[k] = amplitude * np.sin(np.pi * t)
        return xs, ys

    def make_s_curve_path():
        """S-shaped path that moves up then down"""
        xs = np.zeros(91)
        ys = np.zeros(91)
        xs[0] = 0.0
        ys[0] = 0.0
        xs[90] = -2.0
        ys[90] = 0.0
        for k in range(1, 90):
            t = k / 90.0
            xs[k] = -2.0 * t
            # S-curve: up to 1.0 at t=0.3, down to -0.5 at t=0.7, back to 0
            if t < 0.3:
                ys[k] = 1.0 * np.sin(np.pi * t / 0.3)
            elif t < 0.7:
                s = (t - 0.3) / 0.4
                ys[k] = 1.0 - 1.5 * s
            else:
                s = (t - 0.7) / 0.3
                ys[k] = -0.5 * (1 - s)
        return xs, ys

    paths = [
        make_linear_path(),
        make_arc_path(1.5, 1.0),
        make_arc_path(2.0, 1.0),
        make_arc_path(1.0, 0.5),
        make_arc_path(0.8, 1.0),
        make_arc_path(1.2, 1.0),
        make_s_curve_path(),
    ]
    
    best_xs = None
    best_ys = None
    best_score = -1.0
    
    for xs_init, ys_init in paths:
        if time.time() - t_start > time_limit - margin:
            break
        s = compute_score_fast(xs_init, ys_init)
        if s > best_score:
            best_score = s
            best_xs = xs_init.copy()
            best_ys = ys_init.copy()
    
    if best_xs is None:
        best_xs, best_ys = paths[0]
        best_score = compute_score_fast(best_xs, best_ys)

    def refine(xs, ys, steps, max_time=None, use_fast=True):
        """
        Coordinate descent refinement.
        """
        if max_time is None:
            max_time = time_limit - margin
        score_func = compute_score_fast if use_fast else compute_score
        improved = True
        iteration = 0
        while improved and time.time() - t_start < max_time:
            improved = False
            iteration += 1
            for k in range(1, 90):
                if time.time() - t_start > max_time:
                    break
                cx = xs[k]
                cy = ys[k]
                best_cx = cx
                best_cy = cy
                best_s = score_func(xs, ys)
                
                # Try perturbations in x
                for dx in steps:
                    xs[k] = cx + dx
                    s = score_func(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx + dx
                        best_cy = cy
                        improved = True
                # Try perturbations in y
                for dy in steps:
                    xs[k] = cx
                    ys[k] = cy + dy
                    s = score_func(xs, ys)
                    if s > best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy + dy
                        improved = True
                # Try combined (diagonals)
                for dx in steps:
                    for dy in steps:
                        if abs(dx) > 0.15 and abs(dy) > 0.15: 
                            continue
                        xs[k] = cx + dx
                        ys[k] = cy + dy
                        s = score_func(xs, ys)
                        if s > best_s:
                            best_s = s
                            best_cx = cx + dx
                            best_cy = cy + dy
                            improved = True
                
                xs[k] = best_cx
                ys[k] = best_cy
        
        return xs, ys, best_s

    # Coarse Refinement
    coarse_steps = [-0.5, -0.3, -0.15, 0.15, 0.3, 0.5]
    best_xs, best_ys, best_score = refine(best_xs, best_ys, coarse_steps)

    # Medium Refinement
    if time.time() - t_start < time_limit - margin:
        med_steps = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, med_steps)

    # Fine Refinement
    if time.time() - t_start < time_limit - margin:
        fine_steps = [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, fine_steps)

    # Very Fine Refinement
    if time.time() - t_start < time_limit - margin:
        vfine_steps = [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, vfine_steps)

    # Ultra Fine Refinement
    if time.time() - t_start < time_limit - margin:
        ufine_steps = [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, ufine_steps)

    # Ultra Ultra Fine Refinement
    if time.time() - t_start < time_limit - margin:
        uufine_steps = [-0.01, -0.005, -0.002, 0.002, 0.005, 0.01]
        best_xs, best_ys, best_score = refine(best_xs, best_ys, uufine_steps)

    # Random Restart / Perturbation if time remains
    if time.time() - t_start < time_limit - 80:
        rng = np.random.default_rng(42)
        for restart in range(8):
            if time.time() - t_start > time_limit - margin:
                break
            # Perturb the best path with varying noise levels
            noise_level = 0.15 * (1.0 - restart * 0.1)
            pert_xs = best_xs + rng.normal(0, noise_level, 91)
            pert_ys = best_ys + rng.normal(0, noise_level, 91)
            # Keep endpoints fixed
            pert_xs[0] = 0.0
            pert_ys[0] = 0.0
            pert_xs[90] = -2.0
            pert_ys[90] = 0.0
            
            # Quick refine with moderate steps
            pert_xs, pert_ys, pert_score = refine(pert_xs, pert_ys, [-0.1, 0.1])
            
            if pert_score > best_score:
                best_score = pert_score
                best_xs = pert_xs.copy()
                best_ys = pert_ys.copy()

    # Final check
    if time.time() - t_start < time_limit - margin:
        final_score = compute_score_fast(best_xs, best_ys)
        if final_score > best_score:
            best_score = final_score

    # Construct output
    path_pts = np.column_stack((best_xs, best_ys, a_vals))
    
    # Ensure finite
    if not np.all(np.isfinite(path_pts)):
        path_pts = np.column_stack((np.linspace(0, -2, 91), np.zeros(91), a_vals))
    
    return path_pts
