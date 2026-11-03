import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _grow_and_relax(w, h, deadline, seed=0):
    """
    Grow-and-relax algorithm:
    1. Start with a grid arrangement of small circles
    2. Iteratively grow all radii by a small factor
    3. Run relaxation steps to separate overlapping circles
    4. Project back into bounds
    5. Return the final configuration
    """
    rng = np.random.RandomState(seed)
    
    # Initial grid placement
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    
    pos = np.zeros((N, 2))
    for i in range(N):
        col = i % cols
        row = i // cols
        pos[i, 0] = (col + 0.5) * w / cols
        pos[i, 1] = (row + 0.5) * h / rows
    
    # Start with small radii
    max_r = min(w, h) / 2.0
    r = np.full(N, min(w / cols, h / rows) * 0.3)
    
    # Relaxation parameters
    n_relax_steps = 200
    growth_factor = 1.002
    max_growth_iters = 500
    
    for grow_iter in range(max_growth_iters):
        if time.time() > deadline:
            break
        
        # Try to grow radii
        new_r = r * growth_factor
        # Check if any radius exceeds max
        if np.any(new_r > max_r):
            new_r = np.minimum(new_r, max_r)
        
        # Run relaxation to separate circles
        pos_relaxed = pos.copy()
        r_current = new_r.copy()
        
        for step in range(n_relax_steps):
            if time.time() > deadline:
                break
            
            # Compute forces from overlaps
            forces = np.zeros_like(pos_relaxed)
            overlap_found = False
            
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos_relaxed[i, 0] - pos_relaxed[j, 0]
                    dy = pos_relaxed[i, 1] - pos_relaxed[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    r_sum = r_current[i] + r_current[j]
                    
                    if d < r_sum and d > 1e-10:
                        # Push apart
                        overlap = r_sum - d
                        nx = dx / d
                        ny = dy / d
                        # Proportional to radius (larger circles move less)
                        total_r = r_current[i] + r_current[j]
                        wi = r_current[j] / total_r
                        wj = r_current[i] / total_r
                        forces[i, 0] += wi * nx * overlap
                        forces[i, 1] += wi * ny * overlap
                        forces[j, 0] -= wj * nx * overlap
                        forces[j, 1] -= wj * ny * overlap
                        overlap_found = True
                    elif d < r_sum:
                        # Very close, add small random push
                        forces[i, 0] += rng.uniform(-0.001, 0.001)
                        forces[i, 1] += rng.uniform(-0.001, 0.001)
                        forces[j, 0] += rng.uniform(-0.001, 0.001)
                        forces[j, 1] += rng.uniform(-0.001, 0.001)
                        overlap_found = True
            
            if not overlap_found:
                break
            
            # Apply forces with damping
            damping = 0.5
            pos_relaxed += damping * forces
            
            # Project into bounds: for each circle, ensure it stays within [r_i, w-r_i] x [r_i, h-r_i]
            for i in range(N):
                ri = r_current[i]
                pos_relaxed[i, 0] = np.clip(pos_relaxed[i, 0], ri, w - ri)
                pos_relaxed[i, 1] = np.clip(pos_relaxed[i, 1], ri, h - ri)
        
        # Check if the relaxed configuration is valid
        valid = True
        for i in range(N):
            ri = r_current[i]
            if pos_relaxed[i, 0] < ri - 1e-9 or pos_relaxed[i, 0] > w - ri + 1e-9:
                valid = False
                break
            if pos_relaxed[i, 1] < ri - 1e-9 or pos_relaxed[i, 1] > h - ri + 1e-9:
                valid = False
                break
            if ri < 1e-9:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = pos_relaxed[i, 0] - pos_relaxed[j, 0]
                    dy = pos_relaxed[i, 1] - pos_relaxed[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < r_current[i] + r_current[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            pos = pos_relaxed
            r = r_current
        else:
            # Can't grow further, break
            break
        
        # Slow down growth as we approach limit
        if grow_iter > 200:
            growth_factor = 1.001
        elif grow_iter > 400:
            growth_factor = 1.0005
    
    return pos, r

def _polish(pos, radii, w, h, deadline):
    """
    Polish the configuration using scipy optimization.
    """
    def objective(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        obj = -np.sum(r)
        
        # Wall violations
        v_x_min = np.maximum(0, r - x)
        v_x_max = np.maximum(0, r - (w - x))
        v_y_min = np.maximum(0, r - y)
        v_y_max = np.maximum(0, r - (h - y))
        obj += 50.0 * np.sum(v_x_min**2 + v_x_max**2 + v_y_min**2 + v_y_max**2)
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy + 1e-12)
                r_sum = r[i] + r[j]
                if d < r_sum:
                    v = r_sum - d
                    obj += 50.0 * v*v
        
        return obj
    
    z0 = np.zeros(3 * N)
    z0[0::3] = pos[:, 0]
    z0[1::3] = pos[:, 1]
    z0[2::3] = radii
    
    r_max = min(w, h) / 2.0
    bounds = []
    for i in range(N):
        bounds.append((0.0, w))
        bounds.append((0.0, h))
        bounds.append((0.0, r_max))
    
    try:
        res = minimize(objective, z0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False})
        
        if res.fun < 0:
            z_opt = res.x
            pos_opt = np.column_stack((z_opt[0::3], z_opt[1::3]))
            r_opt = z_opt[2::3]
            
            # Validate
            valid = True
            for i in range(N):
                ri = r_opt[i]
                if ri < -1e-9:
                    valid = False
                    break
                if pos_opt[i, 0] < ri - 1e-7 or pos_opt[i, 0] > w - ri + 1e-7:
                    valid = False
                    break
                if pos_opt[i, 1] < ri - 1e-7 or pos_opt[i, 1] > h - ri + 1e-7:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos_opt[i, 0] - pos_opt[j, 0]
                        dy = pos_opt[i, 1] - pos_opt[j, 1]
                        d = np.sqrt(dx*dx + dy*dy)
                        if d < r_opt[i] + r_opt[j] - 1e-7:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                return pos_opt, r_opt, np.sum(r_opt)
    except Exception:
        pass
    
    return pos, radii, np.sum(radii)

def _pack_for_aspect(w, h, deadline, num_trials=3):
    """
    Pack circles in w x h rectangle using grow-and-relax + polish.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        pos, radii = _grow_and_relax(w, h, deadline, seed=trial * 1000)
        
        if np.sum(radii) > best_score:
            best_score = np.sum(radii)
            best_pos = pos.copy()
            best_radii = radii.copy()
    
    # Polish the best result
    if best_pos is not None and time.time() < deadline - 10:
        pos_pol, rad_pol, score_pol = _polish(best_pos, best_radii, w, h, deadline)
        if score_pol > best_score:
            best_score = score_pol
            best_pos = pos_pol
            best_radii = rad_pol
    
    return best_pos, best_radii, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    # Perimeter 4 => 2(w+h)=4 => w+h=2
    # w = 2*ar/(1+ar), h = 2/(1+ar)
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        pos, radii, score = _pack_for_aspect(w, h, deadline, num_trials=2)
        
        if pos is None:
            continue
        
        if score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
    
    # Fallback if no valid result
    if best_global_result is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        pos = np.zeros((N, 2))
        rad = np.zeros(N)
        r_val = 0.5 / side * 0.8
        for i in range(N):
            col = i % side
            row = i // side
            pos[i, 0] = (col + 0.5) / side
            pos[i, 1] = (row + 0.5) / side
            rad[i] = r_val
        best_global_result = np.column_stack([pos, rad])
    
    return best_global_result
