import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _validate(pos, radii, w, h, tol=1e-7):
    """Check if configuration is valid: no overlaps, within bounds."""
    if pos.shape != (N, 2) or radii.shape != (N,):
        return False
    for i in range(N):
        ri = radii[i]
        if ri < -tol:
            return False
        if pos[i, 0] < ri - tol or pos[i, 0] > w - ri + tol:
            return False
        if pos[i, 1] < ri - tol or pos[i, 1] > h - ri + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _hexagonal_positions(w, h, scale=1.0):
    """
    Generate hexagonal lattice positions scaled to fit in w x h.
    Returns positions and a base radius for the lattice.
    """
    # Hexagonal packing: rows offset by half spacing
    # For N circles, find arrangement that fits
    # Try different numbers of columns
    best_pos = None
    best_r = 0
    
    for cols in range(3, 12):
        rows = int(np.ceil(N / cols))
        
        # Hexagonal spacing
        # Horizontal spacing = 2*r, vertical spacing = sqrt(3)*r
        # For cols in a row, width needed = (cols-1)*2r + 2r = cols*2r
        # For rows, height needed = (rows-1)*sqrt(3)*r + 2r
        
        # Fit in rectangle
        # cols*2r <= w => r <= w/(2*cols)
        # (rows-1)*sqrt(3)*r + 2r <= h => r*(2 + (rows-1)*sqrt(3)) <= h => r <= h/(2 + (rows-1)*sqrt(3))
        
        r_max = min(w / (2 * cols), h / (2 + (rows - 1) * np.sqrt(3)))
        
        if r_max <= 0:
            continue
        
        # Place circles
        pos = np.zeros((N, 2))
        count = 0
        for row in range(rows):
            # Alternate rows offset
            offset = 0.0 if row % 2 == 0 else r_max
            for col in range(cols):
                if count >= N:
                    break
                x = (col + 0.5) * 2 * r_max + offset
                y = (row + 0.5) * np.sqrt(3) * r_max + r_max
                
                # Adjust to fit in bounds
                x = np.clip(x, r_max, w - r_max)
                y = np.clip(y, r_max, h - r_max)
                
                pos[count, 0] = x
                pos[count, 1] = y
                count += 1
        
        if count < N:
            continue
        
        # Check if this arrangement is valid with r_max
        pos = pos[:N]
        if _validate(pos, np.full(N, r_max), w, h):
            if r_max > best_r:
                best_r = r_max
                best_pos = pos.copy()
    
    if best_pos is not None:
        return best_pos, best_r
    
    # Fallback: grid
    cols = int(np.ceil(np.sqrt(N * w / h)))
    rows = int(np.ceil(N / cols))
    pos = np.zeros((N, 2))
    r = min(w / cols, h / rows) / 2 * 0.9
    for i in range(N):
        col = i % cols
        row = i // cols
        pos[i, 0] = (col + 0.5) * w / cols
        pos[i, 1] = (row + 0.5) * h / rows
    return pos, r

def _optimize_positions(pos0, radii, w, h, deadline):
    """
    Optimize positions only, with fixed radii.
    Maximize minimum clearance (or equivalently, minimize overlap penalty).
    """
    def objective(p):
        x = p[0::2]
        y = p[1::2]
        
        # Wall violations
        v_x_min = np.maximum(0, radii - x)
        v_x_max = np.maximum(0, radii - (w - x))
        v_y_min = np.maximum(0, radii - y)
        v_y_max = np.maximum(0, radii - (h - y))
        wall_pen = np.sum(v_x_min**2 + v_x_max**2 + v_y_min**2 + v_y_max**2)
        
        # Overlap violations - vectorized
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy + 1e-12)
                r_sum = radii[i] + radii[j]
                if d < r_sum:
                    v = r_sum - d
                    overlap_pen += v * v
        
        return 100.0 * wall_pen + 100.0 * overlap_pen
    
    p0 = np.zeros(2 * N)
    p0[0::2] = pos0[:, 0]
    p0[1::2] = pos0[:, 1]
    
    bounds = []
    for i in range(N):
        bounds.append((radii[i], w - radii[i]))
        bounds.append((radii[i], h - radii[i]))
    
    try:
        res = minimize(objective, p0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
        
        if res.fun < 1e-10:
            pos_opt = np.column_stack((res.x[0::2], res.x[1::2]))
            if _validate(pos_opt, radii, w, h):
                return pos_opt
    except Exception:
        pass
    
    return pos0

def _pack_fixed_radii(w, h, deadline, num_trials=3):
    """
    Pack circles with fixed radii derived from hexagonal lattice.
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Get hexagonal initial positions and base radius
        pos_init, r_base = _hexagonal_positions(w, h)
        
        if r_base <= 0:
            continue
        
        # Try to scale up the radius
        # Start with r_base and try to increase
        r_current = r_base
        pos_current = pos_init.copy()
        
        # Try to grow radii
        for grow_iter in range(100):
            if time.time() > deadline - 10:
                break
            
            # Try to grow
            r_try = r_current * 1.005
            
            # Check if all circles can fit with this radius
            # Simple check: each circle needs space
            feasible = True
            for i in range(N):
                if pos_init[i, 0] < r_try or pos_init[i, 0] > w - r_try:
                    feasible = False
                    break
                if pos_init[i, 1] < r_try or pos_init[i, 1] > h - r_try:
                    feasible = False
                    break
            
            if not feasible:
                break
            
            # Optimize positions with new radius
            pos_opt = _optimize_positions(pos_init, np.full(N, r_try), w, h, deadline)
            
            if pos_opt is not None and _validate(pos_opt, np.full(N, r_try), w, h):
                r_current = r_try
                pos_current = pos_opt
            else:
                break
        
        score = np.sum(np.full(N, r_current))
        if score > best_score:
            best_score = score
            best_pos = pos_current.copy()
            best_radii = np.full(N, r_current)
    
    if best_pos is not None:
        return best_pos, best_radii, best_score
    return None, None, 0

def _pack_joint(w, h, deadline, num_trials=2):
    """
    Joint position-radius optimization (fallback).
    """
    best_score = 0
    best_pos = None
    best_radii = None
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        # Grid initialization
        cols = int(np.ceil(np.sqrt(N * w / h)))
        rows = int(np.ceil(N / cols))
        pos = np.zeros((N, 2))
        for i in range(N):
            col = i % cols
            row = i // cols
            pos[i, 0] = (col + 0.5) * w / cols
            pos[i, 1] = (row + 0.5) * h / rows
        
        r = np.full(N, min(w / cols, h / rows) * 0.3)
        
        # Grow and relax
        for grow_iter in range(300):
            if time.time() > deadline - 10:
                break
            
            r_new = r * 1.003
            r_new = np.minimum(r_new, min(w, h) / 2.0)
            
            pos_relaxed = pos.copy()
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                
                forces = np.zeros_like(pos_relaxed)
                overlap_found = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = pos_relaxed[i, 0] - pos_relaxed[j, 0]
                        dy = pos_relaxed[i, 1] - pos_relaxed[j, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        r_sum = r_new[i] + r_new[j]
                        if d < r_sum and d > 1e-10:
                            overlap = r_sum - d
                            nx = dx / d
                            ny = dy / d
                            total_r = r_new[i] + r_new[j]
                            wi = r_new[j] / total_r
                            wj = r_new[i] / total_r
                            forces[i, 0] += wi * nx * overlap
                            forces[i, 1] += wi * ny * overlap
                            forces[j, 0] -= wj * nx * overlap
                            forces[j, 1] -= wj * ny * overlap
                            overlap_found = True
                        elif d < r_sum:
                            forces[i, 0] += np.random.uniform(-0.001, 0.001)
                            forces[i, 1] += np.random.uniform(-0.001, 0.001)
                            forces[j, 0] += np.random.uniform(-0.001, 0.001)
                            forces[j, 1] += np.random.uniform(-0.001, 0.001)
                            overlap_found = True
                
                if not overlap_found:
                    break
                
                pos_relaxed += 0.5 * forces
                for i in range(N):
                    ri = r_new[i]
                    pos_relaxed[i, 0] = np.clip(pos_relaxed[i, 0], ri, w - ri)
                    pos_relaxed[i, 1] = np.clip(pos_relaxed[i, 1], ri, h - ri)
            
            # Validate
            valid = _validate(pos_relaxed, r_new, w, h)
            if valid:
                pos = pos_relaxed
                r = r_new
            else:
                break
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_radii = r.copy()
    
    if best_pos is not None:
        return best_pos, best_radii, best_score
    return None, None, 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_global_score = 0
    best_global_result = None
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.66, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7, 1.8, 0.55, 2.2]
    
    for ar in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        w = 2 * ar / (1 + ar)
        h = 2 / (1 + ar)
        
        # Try fixed-radius approach first (faster, more robust)
        pos, radii, score = _pack_fixed_radii(w, h, deadline, num_trials=2)
        
        if pos is not None and score > best_global_score:
            best_global_score = score
            best_global_result = np.column_stack([pos, radii])
        
        # Try joint optimization if time allows
        if time.time() < deadline - 20:
            pos2, radii2, score2 = _pack_joint(w, h, deadline, num_trials=1)
            if pos2 is not None and score2 > best_global_score:
                best_global_score = score2
                best_global_result = np.column_stack([pos2, radii2])
    
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
