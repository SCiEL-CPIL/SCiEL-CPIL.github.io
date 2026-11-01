import numpy as np
import time
import math

N = 21
DEADLINE_PAD = 10.0

def construct():
    t0 = time.time()
    deadline = t0 + 120.0 - DEADLINE_PAD
    
    best_val = 0.0
    best_config = None
    
    # Try multiple random restarts
    # We can also try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.2, 0.7, 1.4, 0.6, 1.5]
    
    for w0 in aspect_ratios:
        if time.time() > deadline - 30.0:
            break
        h0 = 2.0 - w0
        if h0 < 0.4:
            continue
            
        # Run SA for this aspect ratio
        x, y, r, val = _simulated_annealing(w0, h0, deadline)
        
        if val > best_val:
            best_val = val
            best_config = (x, y, r, w0, h0)
            
    if best_config is None:
        # Fallback
        w0, h0 = 1.0, 1.0
        x, y = _grid_init(w0, h0)
        r = np.full(N, 0.01)
        return np.column_stack((x, y, r))
        
    x, y, r, w0, h0 = best_config
    
    # Final safety check and shrink
    for _ in range(50):
        if _is_valid(x, y, r * (1 - 1e-6), w0, h0):
            r = r * (1 - 1e-6)
        else:
            break
            
    return np.column_stack((x, y, r))

def _grid_init(w, h):
    side = int(np.ceil(np.sqrt(N)))
    x = np.empty(N)
    y = np.empty(N)
    for i in range(N):
        row, col = divmod(i, side)
        x[i] = (col + 0.5) / side * w
        y[i] = (row + 0.5) / side * h
    return x, y

def _compute_max_radii(x, y, w, h):
    """
    Given positions x, y, compute the maximum radius for each circle
    such that all circles are disjoint and inside the rectangle.
    This is the same logic as the seed's Phase A but vectorized/optimized.
    """
    r = np.zeros(N)
    # Initialize with wall distances
    for i in range(N):
        r[i] = min(x[i], w - x[i], y[i], h - y[i])
        
    # Iteratively enforce pairwise constraints
    # r_i <= d_ij - r_j
    # We can do a few passes of relaxation
    for _ in range(50):
        changed = False
        for i in range(N):
            min_r = r[i]
            for j in range(N):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = math.sqrt(dx*dx + dy*dy)
                # Constraint: r_i + r_j <= d  =>  r_i <= d - r_j
                max_allowed = d - r[j]
                if max_allowed < min_r:
                    min_r = max_allowed
            if min_r < r[i] - 1e-9:
                r[i] = max(0.0, min_r)
                changed = True
        if not changed:
            break
            
    return r

def _simulated_annealing(w, h, deadline):
    # Initialize with a jittered grid
    x, y = _grid_init(w, h)
    # Add some jitter to break symmetry
    jitter = 0.05
    x += np.random.uniform(-jitter, jitter, N)
    y += np.random.uniform(-jitter, jitter, N)
    
    # Clip to valid bounds (with a small margin to ensure initial validity)
    x = np.clip(x, 0.1, w - 0.1)
    y = np.clip(y, 0.1, h - 0.1)
    
    r = _compute_max_radii(x, y, w, h)
    current_val = np.sum(r)
    
    best_x = x.copy()
    best_y = y.copy()
    best_r = r.copy()
    best_val = current_val
    
    # SA Parameters
    T_start = 0.1
    T_end = 0.0001
    n_steps = 2000
    
    # Estimate steps per second to adjust n_steps? 
    # Let's just run until deadline.
    
    step_size = 0.1
    
    while time.time() < deadline - 5.0:
        # Cooling schedule: linear or exponential? 
        # Let's do a simple restart-based approach or continuous cooling.
        # Given the short time, let's do a few short SA runs with restarts.
        
        T = T_start
        # Run one SA chain
        for step in range(n_steps):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random circle to move
            i = np.random.randint(N)
            
            # Save old state
            old_xi = x[i]
            old_yi = y[i]
            old_r = r.copy()
            old_val = current_val
            
            # Propose move: Gaussian noise
            x[i] += np.random.normal(0, step_size)
            y[i] += np.random.normal(0, step_size)
            
            # Clip to walls (soft clip, allowing some invalidity if penalty used, 
            # but here we compute max radii which handles walls. 
            # However, if x[i] is way outside, max radius is 0 or negative.
            # We should keep x,y within [0, w] and [0, h] to avoid huge penalties?
            # Actually, if x[i] < 0, wall distance is negative. 
            # Let's just clip x,y to [0, w] and [0, h] to keep them in the domain.
            x[i] = np.clip(x[i], 0.0, w)
            y[i] = np.clip(y[i], 0.0, h)
            
            # Recompute radii
            new_r = _compute_max_radii(x, y, w, h)
            new_val = np.sum(new_r)
            
            delta = new_val - old_val
            
            # Acceptance
            if delta > 0 or np.random.rand() < math.exp(delta / T):
                r = new_r
                current_val = new_val
                
                if current_val > best_val:
                    best_val = current_val
                    best_x = x.copy()
                    best_y = y.copy()
                    best_r = r.copy()
            else:
                # Revert
                x[i] = old_xi
                y[i] = old_yi
                r = old_r
                current_val = old_val
            
            # Cool down
            T *= 0.9995
            if T < T_end:
                break
                
        # Restart with new jitter if time permits
        if time.time() > deadline - 10.0:
            break
            
        # Re-jitter from best or random?
        # Let's restart from best with slight perturbation
        x = best_x + np.random.normal(0, 0.05, N)
        y = best_y + np.random.normal(0, 0.05, N)
        x = np.clip(x, 0.0, w)
        y = np.clip(y, 0.0, h)
        r = _compute_max_radii(x, y, w, h)
        current_val = np.sum(r)
        
        # Reset temperature
        T_start = 0.05
        
    return best_x, best_y, best_r, best_val

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
