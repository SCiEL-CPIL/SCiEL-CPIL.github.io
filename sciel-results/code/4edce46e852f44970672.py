"""Circle packing in perimeter-4 rectangle: Differential Evolution + Growth Polish."""
import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 21

def _is_valid(cx, cy, r, W, H, tol=1e-7):
    """Check if packing is valid."""
    if W <= 0.01 or H <= 0.01:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True

def _penalty_function(z):
    """
    Objective function for Differential Evolution.
    Minimizes negative sum of radii + penalty for violations.
    z = [w, x1..xN, y1..yN, r1..rN]
    """
    w = z[0]
    h = 2.0 - w
    
    if w <= 0.1 or h <= 0.1:
        return 1e10
        
    x = z[1:1+N]
    y = z[1+N:1+2*N]
    r = z[1+2*N:1+3*N]
    
    # Objective: maximize sum of radii
    obj = -np.sum(r)
    
    penalty = 0.0
    
    # Boundary penalties
    # If circle goes out, penalty proportional to violation
    viol_x_left = np.maximum(0, r - x)
    viol_x_right = np.maximum(0, x + r - w)
    viol_y_bottom = np.maximum(0, r - y)
    viol_y_top = np.maximum(0, y + r - h)
    
    penalty += 100.0 * (np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + 
                        np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2))
    
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist:
                # Penalty proportional to squared overlap depth
                overlap = min_dist - np.sqrt(dist_sq)
                penalty += 100.0 * overlap * overlap
                
    return obj + penalty

def _polish_radii(x, y, r, w, h, time_budget, max_rounds=50):
    """
    Iteratively expand radii until no more improvement possible.
    This is a crucial step because DE tends to find positions where circles
    are close but not necessarily maximally expanded.
    """
    start = time.time()
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    for _ in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(N):
            # Max radius limited by walls
            r_max = min(x[i], w - x[i], y[i], h - y[i])
            
            # Max radius limited by neighbors
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            # Only expand if we can make significant progress
            if r_max > r[i] + 1e-8:
                r[i] = max(r_max, 1e-8)
                improved = True
        
        if not improved:
            break
    
    return x, y, r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_value = 0.0
    best_result = None
    
    def update_best(x, y, r, w, h):
        nonlocal best_value, best_result
        if _is_valid(x, y, r, w, h):
            val = float(np.sum(r))
            if val > best_value:
                best_value = val
                best_result = (x.copy(), y.copy(), r.copy(), w, h)
    
    # Bounds for Differential Evolution
    # w in [0.4, 1.6] (since w+h=2, w cannot be too small or too large for 21 circles)
    # x, y, r bounds depend on w, h, but DE requires fixed bounds.
    # We use loose bounds and rely on penalty.
    # Max possible radius is roughly 1 (if 1 circle filled half the square)
    # Min radius is 0
    
    # z = [w, x1..x21, y1..y21, r1..r21]
    # Length: 1 + 21 + 21 + 21 = 64
    
    bounds = [(0.4, 1.6)]  # w
    # x, y can be anywhere in [0, 2] roughly, but penalized if out of [0, w] or [0, h]
    for _ in range(N):
        bounds.append((0.0, 2.0))  # x_i
    for _ in range(N):
        bounds.append((0.0, 2.0))  # y_i
    for _ in range(N):
        bounds.append((0.0, 1.0))  # r_i
    
    # Run Differential Evolution
    # popsize=30 is a good balance. maxiter=100 might be too slow?
    # Let's try popsize=20, maxiter=50.
    # tol=1e-8
    
    try:
        res = differential_evolution(
            _penalty_function,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=20,
            tol=1e-8,
            polish=False,  # We do our own polish
            init='sobol'
        )
    except Exception:
        pass

    if time.time() < deadline - 10:
        # Extract best solution from DE
        w_opt = res.x[0]
        h_opt = 2.0 - w_opt
        x_opt = res.x[1:1+N]
        y_opt = res.x[1+N:1+2*N]
        r_opt = res.x[1+2*N:1+3*N]
        
        # Clip to valid ranges just in case
        x_opt = np.clip(x_opt, 0, w_opt)
        y_opt = np.clip(y_opt, 0, h_opt)
        r_opt = np.clip(r_opt, 0, min(w_opt, h_opt)/2)
        
        # Polish radii
        x_opt, y_opt, r_opt = _polish_radii(x_opt, y_opt, r_opt, w_opt, h_opt, time_budget=10)
        
        # Check validity and update
        update_best(x_opt, y_opt, r_opt, w_opt, h_opt)
        
        # If DE result was bad, try a few random restarts or simpler heuristics?
        # Let's try a simple hexagonal init + polish as a fallback if DE failed or was slow
        if best_value < 1.0:
             # Fallback: Hexagonal grid
             w, h = 1.0, 1.0
             # Simple grid init
             cols = 5
             rows = 5
             # Adjust for 21 circles
             # 5x5 = 25, remove 4 corners?
             # Or just place them randomly and polish
             random.seed(0)
             np.random.seed(0)
             x_init = np.random.uniform(0.1, 0.9, N)
             y_init = np.random.uniform(0.1, 0.9, N)
             r_init = np.full(N, 0.05)
             
             # Force resolve to make valid
             # Simple iterative relaxation
             for _ in range(100):
                 for i in range(N):
                     # Push away from walls
                     if x_init[i] < r_init[i]: x_init[i] = r_init[i]
                     if x_init[i] > w - r_init[i]: x_init[i] = w - r_init[i]
                     if y_init[i] < r_init[i]: y_init[i] = r_init[i]
                     if y_init[i] > h - r_init[i]: y_init[i] = h - r_init[i]
                     
                     # Push away from other circles
                     for j in range(N):
                         if i == j: continue
                         dx = x_init[i] - x_init[j]
                         dy = y_init[i] - y_init[j]
                         d = np.sqrt(dx*dx + dy*dy)
                         min_d = r_init[i] + r_init[j]
                         if d < min_d:
                             if d < 1e-6:
                                 dx, dy = 0.01, 0.01
                                 d = np.sqrt(dx*dx + dy*dy)
                             push = (min_d - d) / 2.0
                             x_init[i] += push * dx / d
                             y_init[i] += push * dy / d
                             x_init[j] -= push * dx / d
                             y_init[j] -= push * dy / d
                             
             x_init = np.clip(x_init, r_init, w - r_init)
             y_init = np.clip(y_init, r_init, h - r_init)
             
             x_init, y_init, r_init = _polish_radii(x_init, y_init, r_init, w, h, time_budget=5)
             update_best(x_init, y_init, r_init, w, h)
             
             # Try a few other aspect ratios with random init + polish
             for w_test in [0.8, 1.2, 0.7, 1.3]:
                 if time.time() > deadline - 5:
                     break
                 h_test = 2.0 - w_test
                 np.random.seed(int(w_test*100))
                 x_test = np.random.uniform(0.1, w_test-0.1, N)
                 y_test = np.random.uniform(0.1, h_test-0.1, N)
                 r_test = np.full(N, 0.05)
                 
                 # Quick relaxation
                 for _ in range(50):
                     for i in range(N):
                         x_test[i] = np.clip(x_test[i], r_test[i], w_test - r_test[i])
                         y_test[i] = np.clip(y_test[i], r_test[i], h_test - r_test[i])
                         for j in range(N):
                             if i == j: continue
                             dx = x_test[i] - x_test[j]
                             dy = y_test[i] - y_test[j]
                             d = np.sqrt(dx*dx + dy*dy)
                             min_d = r_test[i] + r_test[j]
                             if d < min_d:
                                 if d < 1e-6:
                                     dx, dy = 0.01, 0.01
                                     d = np.sqrt(dx*dx + dy*dy)
                                 push = (min_d - d) / 2.0
                                 x_test[i] += push * dx / d
                                 y_test[i] += push * dy / d
                                 x_test[j] -= push * dx / d
                                 y_test[j] -= push * dy / d
                                 
                 x_test = np.clip(x_test, r_test, w_test - r_test)
                 y_test = np.clip(y_test, r_test, h_test - r_test)
                 
                 x_test, y_test, r_test = _polish_radii(x_test, y_test, r_test, w_test, h_test, time_budget=5)
                 update_best(x_test, y_test, r_test, w_test, h_test)

    # Fallback if nothing worked
    if best_result is None:
        w, h = 1.0, 1.0
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = np.full(N, 0.05)
        best_result = (x, y, r, w, h)
    
    x, y, r, w, h = best_result
    
    # Final safety shrink
    for _ in range(3):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break
            
    return np.column_stack((x, y, r))
