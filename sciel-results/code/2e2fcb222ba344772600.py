"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _objective(coords, widths, heights):
    """Return (sum_radii, min_clearance, max_half_width, max_half_height)."""
    n = len(widths)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]
    
    # Sum of radii
    sum_r = np.sum(rs)
    
    # Check disjoint interiors: dist > r_i + r_j
    min_clearance = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            clearance = d - (rs[i] + rs[j])
            if clearance < min_clearance:
                min_clearance = clearance
    
    # Check bounding box fits in widths x heights
    half_w = np.max(xs + rs) - np.min(xs - rs)
    half_h = np.max(ys + rs) - np.min(ys - rs)
    
    # We need half_w <= widths/2 and half_h <= heights/2
    # Actually widths and heights are full dimensions
    # Let's redefine: the rectangle has width W and height H, perimeter 2(W+H)=4 => W+H=2
    # Circles must fit inside [0,W]x[0,H]
    # So x_i - r_i >= 0, x_i + r_i <= W, y_i - r_i >= 0, y_i + r_i <= H
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    W_needed = max_x - min_x
    H_needed = max_y - min_y
    
    # We want to maximize sum_r subject to W_needed + H_needed <= 2
    # For scoring, we return the raw value which is sum_r
    # But we need to ensure validity. The evaluator checks validity.
    
    return sum_r, min_clearance, W_needed, H_needed


def _is_valid(coords, eps=1e-6):
    """Check if the configuration is valid."""
    n = len(coords)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]
    
    # Check radii positive
    if np.any(rs < -eps):
        return False
    
    # Check disjoint interiors
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < rs[i] + rs[j] - eps:
                return False
    
    # Check bounding box perimeter
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    W = max_x - min_x
    H = max_y - min_y
    perimeter = 2 * (W + H)
    
    if perimeter > 4 + 1e-6:
        return False
    
    return True


def _scale_to_fit(coords, target_perimeter=4.0, margin=1e-4):
    """Scale the configuration so that its bounding box perimeter is exactly target_perimeter."""
    n = len(coords)
    xs = coords[:, 0]
    ys = coords[:, 1]
    rs = coords[:, 2]
    
    min_x = np.min(xs - rs)
    max_x = np.max(xs + rs)
    min_y = np.min(ys - rs)
    max_y = np.max(ys + rs)
    
    W = max_x - min_x
    H = max_y - min_y
    
    if W + H == 0:
        return coords.copy()
    
    # We want 2(W' + H') = target_perimeter => W' + H' = target_perimeter/2
    # Scale factor s: s*(W + H) = target_perimeter/2 => s = target_perimeter / (2*(W+H))
    s = (target_perimeter / 2.0) / (W + H)
    
    # Apply scaling: multiply all coordinates and radii by s, then translate to [0, W']x[0, H']
    xs_scaled = xs * s
    ys_scaled = ys * s
    rs_scaled = rs * s
    
    # Translate so min is at 0
    new_min_x = np.min(xs_scaled - rs_scaled)
    new_min_y = np.min(ys_scaled - rs_scaled)
    
    xs_final = xs_scaled - new_min_x
    ys_final = ys_scaled - new_min_y
    
    result = np.column_stack([xs_final, ys_final, rs_scaled])
    
    # Add margin to avoid floating point issues
    # Actually, we should shrink slightly to be safe
    # But scaling to exactly target_perimeter might cause validity issues due to FP
    # Let's scale to target_perimeter * (1 - margin)
    return result


def _initial_grid():
    """Create an initial grid-like configuration."""
    # Use a 5x5 grid, take 21 circles
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
    
    return np.array(cells[:N])


def _local_optimize(coords, max_iter=1000, step_size=0.01, cooling=0.995):
    """Simple local optimization using gradient-free method."""
    best_coords = coords.copy()
    best_sum_r, _, _, _ = _objective(best_coords)
    current_sum_r = best_sum_r
    
    temp = step_size
    for it in range(max_iter):
        # Random perturbation
        delta = np.random.normal(0, temp, size=coords.shape)
        new_coords = coords + delta
        
        # Ensure radii are positive
        new_coords[:, 2] = np.maximum(new_coords[:, 2], 1e-6)
        
        new_sum_r, min_clear, W, H = _objective(new_coords)
        
        # Check validity
        if min_clear < -1e-4 or W + H > 2.0 + 1e-4:
            # Invalid, reject
            if np.random.random() < math.exp(-10.0 / max(temp, 1e-8)):
                coords = new_coords
            continue
        
        # Accept if better or with probability
        if new_sum_r > current_sum_r:
            coords = new_coords
            current_sum_r = new_sum_r
            if current_sum_r > best_sum_r:
                best_coords = coords.copy()
                best_sum_r = current_sum_r
        else:
            # Metropolis acceptance
            delta_obj = current_sum_r - new_sum_r
            if np.random.random() < math.exp(-delta_obj / max(temp, 1e-8)):
                coords = new_coords
                current_sum_r = new_sum_r
        
        temp *= cooling
    
    return best_coords


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with grid
    coords = _initial_grid()
    
    # Scale to fit perimeter 4
    coords = _scale_to_fit(coords, target_perimeter=4.0)
    
    # Local optimization
    best_coords = coords.copy()
    best_sum_r, _, _, _ = _objective(best_coords)
    
    # Multiple restarts
    n_restarts = 10
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb initial solution
        if r == 0:
            curr = best_coords.copy()
        else:
            # Create a new random configuration
            # Use a jittered grid
            side = 5
            cells = []
            for i in range(side):
                for j in range(side):
                    if len(cells) < N:
                        x = (i + 0.5) / side + np.random.uniform(-0.1, 0.1) / side
                        y = (j + 0.5) / side + np.random.uniform(-0.1, 0.1) / side
                        r = 0.5 / side - 1e-4 + np.random.uniform(-0.05, 0.05) / side
                        cells.append([x, y, max(r, 1e-4)])
            curr = np.array(cells[:N])
            curr = _scale_to_fit(curr, target_perimeter=4.0)
        
        # Local optimization
        opt_coords = _local_optimize(curr, max_iter=500, step_size=0.05, cooling=0.99)
        
        # Scale to fit exactly
        opt_coords = _scale_to_fit(opt_coords, target_perimeter=4.0)
        
        sum_r, min_clear, W, H = _objective(opt_coords)
        
        if min_clear >= -1e-6 and W + H <= 2.0 + 1e-6:
            if sum_r > best_sum_r:
                best_coords = opt_coords.copy()
                best_sum_r = sum_r
    
    # Final polish: try to increase radii slightly
    if time.time() < deadline:
        # Try to scale up slightly
        for factor in [1.001, 1.002, 1.005, 1.01]:
            scaled = best_coords.copy()
            scaled[:, 2] *= factor
            # Check validity
            min_x = np.min(scaled[:, 0] - scaled[:, 2])
            max_x = np.max(scaled[:, 0] + scaled[:, 2])
            min_y = np.min(scaled[:, 1] - scaled[:, 2])
            max_y = np.max(scaled[:, 1] + scaled[:, 2])
            W = max_x - min_x
            H = max_y - min_y
            if 2 * (W + H) <= 4.0 + 1e-6:
                # Check disjoint
                valid = True
                for i in range(N):
                    for j in range(i + 1, N):
                        d = math.hypot(scaled[i, 0] - scaled[j, 0], scaled[i, 1] - scaled[j, 1])
                        if d < scaled[i, 2] + scaled[j, 2] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_coords = scaled
    
    # Ensure final validity
    if not _is_valid(best_coords, eps=1e-5):
        # Fall back to grid
        best_coords = _initial_grid()
        best_coords = _scale_to_fit(best_coords, target_perimeter=4.0)
    
    return best_coords
# EVOLVE-BLOCK-END
