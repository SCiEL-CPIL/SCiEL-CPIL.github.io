"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 21

# EVOLVE-BLOCK-START

def _validate(arr):
    """Check if arrangement is valid: perimeter <= 4, non-overlapping, positive radii."""
    if arr is None or len(arr) != N:
        return False
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    if np.any(rs <= 0):
        return False
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if 2 * (w + h) > 4.0 + 1e-9:
        return False
    # Check pairwise distances
    dists = cdist(arr[:, :2], arr[:, :2])
    min_seps = rs[:, None] + rs[None, :]
    if np.any(dists < min_seps - 1e-9):
        return False
    return True


def _score(arr):
    """Return sum of radii if valid, else -inf."""
    if not _validate(arr):
        return -np.inf
    return float(np.sum(arr[:, 2]))


def _initial_grid(n):
    """Create a simple grid-like initial arrangement."""
    side = int(np.ceil(np.sqrt(n)))
    cols = side
    rows = int(np.ceil(n / cols))
    W, H = 2.0, 2.0  # Start with square 2x2 (perimeter 8, will be scaled)
    # Actually, let's use a more compact initial guess
    # Use a rectangle with aspect ratio ~1 for simplicity
    # We'll place circles in a grid and then optimize
    xs = []
    ys = []
    rs = []
    for i in range(n):
        col = i % cols
        row = i // cols
        x = (col + 0.5) / cols * 1.8
        y = (row + 0.5) / rows * 1.8
        r = 0.5 / max(cols, rows) * 0.9
        xs.append(x)
        ys.append(y)
        rs.append(r)
    return np.array([xs, ys, rs]).T


def _objective(arr, W, H):
    """
    Objective: minimize negative sum of radii, with penalties for:
    - Overlap between circles
    - Circles going outside [0, W] x [0, H]
    - Perimeter constraint (should be <= 4)
    """
    xs = arr[:, 0]
    ys = arr[:, 1]
    rs = arr[:, 2]
    
    # Penalty for going outside bounds
    out_x = np.maximum(0, xs - (W - rs)) + np.maximum(0, rs - xs)
    out_y = np.maximum(0, ys - (H - rs)) + np.maximum(0, rs - ys)
    boundary_penalty = np.sum(out_x ** 2) + np.sum(out_y ** 2)
    
    # Penalty for overlap
    dists = cdist(arr[:, :2], arr[:, :2])
    min_seps = rs[:, None] + rs[None, :]
    overlaps = np.maximum(0, min_seps - dists)
    overlap_penalty = np.sum(overlaps ** 2)
    
    # We want to maximize sum of radii, so minimize -sum(rs)
    return -np.sum(rs) + 100.0 * (boundary_penalty + overlap_penalty)


def _optimize_from(start, W, H, time_limit):
    """Run L-BFGS-B optimization from a starting point."""
    start_time = time.time()
    
    # Bounds: x in [0, W], y in [0, H], r in [0.01, min(W, H)/2]
    r_max = min(W, H) / 2.0
    bounds = [(0, W) for _ in range(N)] + [(0, H) for _ in range(N)] + [(0.01, r_max)] * N
    
    result = None
    try:
        result = minimize(
            _objective,
            start,
            args=(W, H),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 50000}
        )
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if result is not None and result.success:
        return result.x, elapsed
    elif result is not None:
        return result.x, elapsed
    return start, elapsed


def _try_aspect_ratios(arr, time_limit):
    """Try several aspect ratios for the bounding rectangle."""
    best_arr = None
    best_score = -np.inf
    start_time = time.time()
    
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    # Try various aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.667, 1.5, 0.5, 2.0, 0.4, 2.5]
    
    for ar in aspect_ratios:
        if time.time() - start_time > time_limit:
            break
        W = 2.0 * ar / (1 + ar)
        H = 2.0 / (1 + ar)
        
        # Scale the initial arrangement to fit in [0, W] x [0, H]
        # First, normalize arr to [0,1] x [0,1]
        xs = arr[:, 0]
        ys = arr[:, 1]
        rs = arr[:, 2]
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        # Scale to fit
        scale_x = W / x_range
        scale_y = H / y_range
        scale = min(scale_x, scale_y) * 0.95
        
        new_xs = (xs - x_min) * scale
        new_ys = (ys - y_min) * scale
        new_rs = rs * scale
        
        # Center in the rectangle
        new_xs += (W - (new_xs.max() - new_xs.min())) / 2
        new_ys += (H - (new_ys.max() - new_ys.min())) / 2
        
        candidate = np.column_stack([new_xs, new_ys, new_rs])
        
        # Optimize
        opt_arr, elapsed = _optimize_from(candidate, W, H, time_limit - (time.time() - start_time))
        
        if opt_arr is not None:
            # Ensure positivity
            opt_arr[:, 2] = np.maximum(opt_arr[:, 2], 0.01)
            s = _score(opt_arr)
            if s > best_score:
                best_score = s
                best_arr = opt_arr.copy()
    
    return best_arr, best_score


def construct():
    """
    Construct an arrangement of 21 circles in a rectangle of perimeter <= 4.
    Returns array of shape (21, 3) with [x, y, r] for each circle.
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with a grid arrangement
    arr = _initial_grid(N)
    
    # Try multiple aspect ratios and optimize
    best_arr, best_score = _try_aspect_ratios(arr, 130.0)
    
    # If we still have time, try a few more restarts with perturbed initial positions
    if best_arr is not None and time.time() < deadline - 20:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            # Perturb the best arrangement slightly
            perturbed = best_arr.copy()
            perturbed[:, 0] += rng.normal(0, 0.05, N)
            perturbed[:, 1] += rng.normal(0, 0.05, N)
            perturbed[:, 2] = np.maximum(perturbed[:, 2] * (1 + rng.normal(0, 0.1, N)), 0.01)
            
            # Try a couple of aspect ratios
            for ar in [1.0, 0.8, 1.25]:
                if time.time() > deadline - 20:
                    break
                W = 2.0 * ar / (1 + ar)
                H = 2.0 / (1 + ar)
                
                # Rescale to fit
                xs = perturbed[:, 0]
                ys = perturbed[:, 1]
                rs = perturbed[:, 2]
                
                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                x_range = x_max - x_min if x_max > x_min else 1.0
                y_range = y_max - y_min if y_max > y_min else 1.0
                
                scale_x = W / x_range
                scale_y = H / y_range
                scale = min(scale_x, scale_y) * 0.95
                
                new_xs = (xs - x_min) * scale + (W - (xs.max() - xs.min()) * scale) / 2
                new_ys = (ys - y_min) * scale + (H - (ys.max() - ys.min()) * scale) / 2
                new_rs = rs * scale
                
                candidate = np.column_stack([new_xs, new_ys, new_rs])
                opt_arr, elapsed = _optimize_from(candidate, W, H, deadline - time.time() - 5)
                
                if opt_arr is not None:
                    opt_arr[:, 2] = np.maximum(opt_arr[:, 2], 0.01)
                    s = _score(opt_arr)
                    if s > best_score:
                        best_score = s
                        best_arr = opt_arr.copy()
    
    # Final validation and cleanup
    if best_arr is None:
        # Fallback: return a valid grid arrangement
        arr = _initial_grid(N)
        xs, ys, rs = arr[:, 0], arr[:, 1], arr[:, 2]
        W, H = 2.0, 2.0
        scale = 0.95
        new_xs = xs * scale
        new_ys = ys * scale
        new_rs = rs * scale
        best_arr = np.column_stack([new_xs, new_ys, new_rs])
    
    # Ensure all radii are positive
    best_arr[:, 2] = np.maximum(best_arr[:, 2], 0.01)
    
    # Final check: if invalid, try to fix by scaling down
    if not _validate(best_arr):
        xs, ys, rs = best_arr[:, 0], best_arr[:, 1], best_arr[:, 2]
        W = xs.max() - xs.min()
        H = ys.max() - ys.min()
        if 2 * (W + H) > 4.0:
            scale = 4.0 / (2 * (W + H)) * 0.99
            best_arr[:, 0] = (xs - xs.min()) * scale
            best_arr[:, 1] = (ys - ys.min()) * scale
            best_arr[:, 2] = rs * scale
    
    return best_arr

# EVOLVE-BLOCK-END
