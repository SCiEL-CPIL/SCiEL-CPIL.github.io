"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
import time
import math

N = 21
PERIMETER = 4.0
TIME_LIMIT = 110.0  # Safety margin before 120s limit


def _validate_and_compute(circles, tol=1e-9):
    """Check validity and compute sum of radii. Returns (valid, value)."""
    if len(circles) == 0:
        return False, 0.0
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False, 0.0
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False, 0.0
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False, 0.0
    return True, np.sum(r)


def _seed_grid():
    """Create a simple grid-based seed configuration."""
    best_val = -1
    best_circles = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        s = 2.0 / (cols + rows)
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * s
                cy = (j + 0.5) * s
                r = 0.5 * s - 1e-4
                circles.append([cx, cy, r])
        circles = np.array(circles[:N])
        
        valid, val = _validate_and_compute(circles)
        if valid and val > best_val:
            best_val = val
            best_circles = circles.copy()
    
    return best_circles


def _optimize(circles, time_budget):
    """Optimize circle positions and radii using gradient-free local search."""
    t_start = time.time()
    best_circles = circles.copy()
    valid, best_val = _validate_and_compute(best_circles)
    
    n = len(circles)
    current = circles.copy()
    step_size = 0.01
    
    while time.time() - t_start < time_budget:
        # Random perturbation
        trial = current.copy()
        for k in range(n):
            for d in range(3):
                if np.random.random() < 0.3:
                    trial[k, d] += np.random.randn() * step_size
        # Ensure positive radii
        trial[:, 2] = np.maximum(trial[:, 2], 1e-6)
        
        valid_trial, val_trial = _validate_and_compute(trial)
        if valid_trial and val_trial > best_val:
            best_val = val_trial
            best_circles = trial.copy()
            current = trial.copy()
            step_size = max(0.001, step_size * 0.95)
        else:
            step_size = max(0.0005, step_size * 0.9)
    
    return best_circles


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    
    # Start with a good seed
    circles = _seed_grid()
    
    if circles is None or len(circles) < N:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        circles = np.array(cells[:N])
        valid, _ = _validate_and_compute(circles)
        if not valid:
             # If even seed is invalid, try smaller radii
             circles[:, 2] *= 0.5
             valid, _ = _validate_and_compute(circles)
             if not valid:
                 circles[:, 2] *= 0.5
    
    # Optimize
    time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
    if time_remaining > 2.0:
        circles = _optimize(circles, time_remaining)
    
    # Final validation
    valid, _ = _validate_and_compute(circles)
    if not valid:
        # If optimization failed to produce a valid result, try to repair or fallback to seed
        # For now, just return the seed if it was valid, else a minimal valid config
        seed_circles = _seed_grid()
        if seed_circles is not None:
            valid_seed, _ = _validate_and_compute(seed_circles)
            if valid_seed:
                circles = seed_circles
        else:
            # Construct a trivially valid small configuration
            s = 2.0 / (7 + 3) # 7x3 grid
            circles = []
            for i in range(7):
                for j in range(3):
                    if len(circles) >= N: break
                    circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
            circles = np.array(circles[:N])
    
    return circles
