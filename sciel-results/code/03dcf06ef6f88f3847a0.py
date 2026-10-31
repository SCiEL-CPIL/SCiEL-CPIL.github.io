"""Circle packing in perimeter-4 rectangle.
Approach: Simulated Annealing (SA) with penalty-based objective.
- We treat the circle positions and radii as a vector to be optimized.
- The objective function maximizes the sum of radii, penalizing overlaps and
  boundary violations quadratically.
- We use a standard Simulated Annealing schedule to escape local minima.
- We try several aspect ratios to find the optimal rectangle shape.
"""
import numpy as np
import time

N = 21
DEADLINE_PAD = 20.0


def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD

    best = None
    best_val = 0.0

    # Aspect ratios to try (w + h = 2)
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6, 0.5, 1.8, 0.9, 1.1]

    for w0 in aspect_ratios:
        if time.time() > deadline - 20:
            break
        h0 = 2.0 - w0
        if h0 < 0.3:
            continue

        # Initialize with a grid
        x, y, r = _init_grid(w0, h0)
        
        # Run Simulated Annealing
        x, y, r = _simulated_anneal(x, y, r, w0, h0, deadline)

        # Check validity and update best
        if _is_valid(x, y, r, w0, h0):
            val = float(np.sum(r))
            if val > best_val:
                best_val = val
                best = (x.copy(), y.copy(), r.copy(), w0, h0)

    if best is None:
        # Fallback
        w0, h0 = 1.0, 1.0
        x, y, r = _init_grid(w0, h0)
        # Shrink to ensure validity
        for _ in range(50):
            if _is_valid(x, y, r * 0.99, w0, h0):
                r *= 0.99
            else:
                break
        return np.column_stack((x, y, r))

    x, y, r, w0, h0 = best
    # Final safety shrink
    for _ in range(20):
        if _is_valid(x, y, r * (1 - 1e-6), w0, h0):
            r = r * (1 - 1e-6)
        else:
            break

    return np.column_stack((x, y, r))


def _init_grid(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    r = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
        r[i] = min(w, h) / (side * 2) * 0.8
    return x, y, r


def _objective(x, y, r, w, h):
    """
    Returns the objective value to be MINIMIZED.
    Objective = -Sum(r) + Penalty(overlaps) + Penalty(walls)
    """
    sum_r = np.sum(r)
    penalty = 0.0
    
    # Wall penalties
    # x < r or x > w-r or y < r or y > h-r
    # We use squared distance for smooth gradients
    wall_pen = 0.0
    for i in range(N):
        # Left
        if x[i] < r[i]:
            wall_pen += (r[i] - x[i]) ** 2
        # Right
        if x[i] > w - r[i]:
            wall_pen += (x[i] - (w - r[i])) ** 2
        # Bottom
        if y[i] < r[i]:
            wall_pen += (r[i] - y[i]) ** 2
        # Top
        if y[i] > h - r[i]:
            wall_pen += (y[i] - (h - r[i])) ** 2
            
    # Overlap penalties
    overlap_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            dist_req = r[i] + r[j]
            if d < dist_req:
                overlap_pen += (dist_req - d) ** 2
                
    # Weights for penalties
    W = 100.0
    return -sum_r + W * (wall_pen + overlap_pen)


def _simulated_anneal(x, y, r, w, h, deadline):
    """
    Simulated Annealing loop.
    """
    x = x.copy().astype(float)
    y = y.copy().astype(float)
    r = r.copy().astype(float)
    
    # Initial temperature
    T = 0.1
    T_min = 1e-5
    cooling = 0.99
    
    # We track the best valid configuration found
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
    best_obj = _objective(x, y, r, w, h)
    
    current_obj = best_obj
    
    # Number of iterations
    max_iters = 10000
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Perturb state
        # Move one circle randomly
        idx = np.random.randint(N)
        
        # Random step size
        step = 0.05 * T + 0.01
        
        # New position
        nx = x[idx] + np.random.uniform(-step, step)
        ny = y[idx] + np.random.uniform(-step, step)
        
        # New radius (slight change)
        nr = r[idx] * (1 + np.random.uniform(-0.05, 0.05))
        nr = max(0.01, min(nr, min(w, h)/2))
        
        # Calculate new objective
        # We only need to update penalties involving idx
        # For simplicity and speed, we recalculate full objective if N is small
        # N=21 is small enough for full recalculation in Python if optimized?
        # Let's try full recalculation first. If too slow, we optimize.
        
        x_new = x.copy()
        y_new = y.copy()
        r_new = r.copy()
        x_new[idx] = nx
        y_new[idx] = ny
        r_new[idx] = nr
        
        new_obj = _objective(x_new, y_new, r_new, w, h)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            x, y, r = x_new, y_new, r_new
            current_obj = new_obj
            
            # Update best if valid (or just best objective)
            # We want to maximize sum of radii, so minimize objective.
            # But we must ensure validity for the final answer.
            # Let's track the best *valid* configuration.
            # Checking validity every step is expensive.
            # Instead, we track the best objective, and at the end, we check validity.
            # Or, we can accept only if penalty is low.
            
            # Heuristic: If penalty is 0, it's valid.
            # _objective returns -sum_r + W*pen.
            # If pen > 0, obj is worse than -sum_r.
            # We can check if the penalty part is close to 0.
            
            # Let's just update best_obj and best_x/y/r
            if new_obj < best_obj:
                best_obj = new_obj
                best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        
        # Cool down
        T *= cooling
        
        # If T is too low, we might be stuck. 
        # We can restart with a higher T if no improvement for a while.
        # For now, simple cooling.
        
    return best_x, best_y, best_r


def _is_valid(x, y, r, w, h, tol=1e-8):
    if w <= 0.05 or h <= 0.05:
        return False
    if np.any(x - r < -tol) or np.any(y - r < -tol):
        return False
    if np.any(x + r > w + tol) or np.any(y + r > h + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True
