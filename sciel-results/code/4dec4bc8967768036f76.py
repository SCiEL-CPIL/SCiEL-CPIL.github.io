"""Circle packing in perimeter-4 rectangle: force-directed + simulated annealing."""
import numpy as np
import time

N = 21
DEADLINE = 140.0  # safety margin before 150s kill

def _perimeter(w, h):
    return 2.0 * (w + h)

def _validity(x, w, h):
    """Check all circles are inside [0,w]x[0,h] and non-overlapping."""
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    if np.any(cx < -1e-9) or np.any(cy < -1e-9) or np.any(cx > w + 1e-9) or np.any(cy > h + 1e-9):
        return False
    if np.any(r < -1e-9):
        return False
    # pairwise distances
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = np.where(np.arange(N)[:, None] != np.arange(N)[None, :], dist, np.inf)
    if np.any(min_dist < np.sum(r[:, None] + r[None, :], axis=1) - 1e-9):
        return False
    return True

def _score(x, w, h):
    """Return (sum_radii, penalty). Higher sum_radii is better; penalty for violations."""
    cx, cy, r = x[:, 0], x[:, 1], x[:, 2]
    sum_r = np.sum(r)
    # boundary penalty
    bound_pen = 0.0
    if np.any(cx < 0) or np.any(cy < 0) or np.any(cx > w) or np.any(cy > h):
        bound_pen = 10.0 * (np.sum(np.maximum(0, -cx)) + np.sum(np.maximum(0, -cy)) +
                            np.sum(np.maximum(0, cx - w)) + np.sum(np.maximum(0, cy - h)))
    if np.any(r < 0):
        bound_pen += 10.0 * np.sum(np.maximum(0, -r))
    # overlap penalty
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    min_dist = np.where(np.arange(N)[:, None] != np.arange(N)[None, :], dist, np.inf)
    overlap = np.maximum(0, np.sum(r[:, None] + r[None, :], axis=1) - min_dist)
    overlap_pen = 10.0 * np.sum(overlap)
    return sum_r, bound_pen + overlap_pen

def _initial_grid(w, h):
    """Create initial grid of N circles in rectangle [0,w]x[0,h]."""
    # Choose grid dimensions close to aspect ratio
    ar = w / h if h > 0 else 1.0
    # Try several grid layouts
    best = None
    for cols in range(1, N + 1):
        rows = int(np.ceil(N / cols))
        if cols * rows < N:
            continue
        cell_w = w / cols
        cell_h = h / rows
        r = 0.45 * min(cell_w, cell_h)
        total_r = N * r
        if best is None or total_r > best[0]:
            best = (total_r, cols, rows, r)
    total_r, cols, rows, r = best
    cx = np.zeros(N)
    cy = np.zeros(N)
    rr = np.full(N, r)
    for i in range(N):
        row = i // cols
        col = i % cols
        cx[i] = (col + 0.5) * w / cols
        cy[i] = (row + 0.5) * h / rows
    return np.column_stack([cx, cy, rr])

def construct():
    start_time = time.time()
    
    # Start with a square-ish rectangle
    w, h = 1.0, 1.0  # perimeter = 4
    
    # Initial configuration
    x = _initial_grid(w, h)
    
    # Track best
    best_x = x.copy()
    best_wh = (w, h)
    best_score, _ = _score(best_x, best_wh[0], best_wh[1])
    best_valid = _validity(best_x, best_wh[0], best_wh[1])
    
    # Simulated annealing
    T = 0.01
    T_min = 1e-7
    cooling = 0.9997
    step_scale = 0.05
    
    # Also anneal the aspect ratio
    log_aspect = 0.0  # w/h = exp(log_aspect), so w = 2*exp(log_aspect)/(1+exp(log_aspect)), h = 2/(1+exp(log_aspect))
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            break
        
        # Update rectangle dimensions from log_aspect
        ea = np.exp(log_aspect)
        w = 2.0 * ea / (1.0 + ea)
        h = 2.0 / (1.0 + ea)
        
        old_score, old_pen = _score(x, w, h)
        
        # Propose new positions
        x_new = x.copy()
        x_new[:, 0] += np.random.randn(N) * step_scale * T
        x_new[:, 1] += np.random.randn(N) * step_scale * T
        x_new[:, 2] += np.random.randn(N) * step_scale * T * 0.5  # radii change slower
        
        # Propose new aspect ratio
        log_aspect_new = log_aspect + np.random.randn() * 0.02 * T
        
        # Clamp radii to positive
        x_new[:, 2] = np.maximum(x_new[:, 2], 1e-4)
        
        # Compute new score with new aspect
        ea_new = np.exp(log_aspect_new)
        w_new = 2.0 * ea_new / (1.0 + ea_new)
        h_new = 2.0 / (1.0 + ea_new)
        
        new_score, new_pen = _score(x_new, w_new, h_new)
        
        # Objective: maximize sum_r - penalty
        old_obj = old_score - old_pen
        new_obj = new_score - new_pen
        delta = new_obj - old_obj
        
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            x = x_new
            log_aspect = log_aspect_new
            
            # Check if valid and better
            if new_pen < 1e-6:
                if new_score > best_score:
                    best_score = new_score
                    best_x = x_new.copy()
                    best_wh = (w_new, h_new)
                    best_valid = True
        
        T *= cooling
        
        # Occasionally re-normalize to keep perimeter exactly 4
        # (already enforced by parametrization)
    
    # Return best valid configuration, or best found
    if best_valid:
        return best_x
    # If no valid config found, return the one with lowest penalty
    return x
