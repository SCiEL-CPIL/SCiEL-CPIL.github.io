import numpy as np
from scipy.optimize import minimize
import time

N = 21

def _sa_pack(w, h, time_limit):
    """
    Simulated Annealing for circle packing.
    """
    n_circles = N
    # Initial grid
    cols = 5
    rows = 5
    # Adjust grid to fit N
    # 21 is 5x5 minus 4 corners? Or 4x5 + 1?
    # Let's just use a jittered grid of 21 points.
    
    x0 = np.zeros(n_circles * 3)
    # Grid positions
    # 21 circles. 5 cols, 5 rows is 25. We take first 21.
    # Or 4 cols, 6 rows?
    # Let's use 5x5 grid, skip last 4.
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= n_circles:
                break
            cx = (i + 0.5) * w / 5.0
            cy = (j + 0.5) * h / 5.0
            x0[2*idx] = cx
            x0[2*idx+1] = cy
            # Small initial radius
            x0[2*idx+2] = 0.01
            idx += 1
            
    # Ensure valid start
    # Check overlaps and shrink if necessary
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            dx = x0[2*i] - x0[2*j]
            dy = x0[2*i+1] - x0[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = dist / 2.0 - 1e-5
            x0[2*i+2] = min(x0[2*i+2], max_r)
            x0[2*j+2] = min(x0[2*j+2], max_r)
            
    # Check walls
    for i in range(n_circles):
        x = x0[2*i]
        y = x0[2*i+1]
        r = x0[2*i+2]
        min_wall = min(x, w-x, y, h-y)
        x0[2*i+2] = min(r, min_wall)
        
    best_x = x0.copy()
    best_score = np.sum(x0[2::2])
    
    start_time = time.time()
    T = 0.1
    T_end = 1e-6
    cooling = 0.999
    
    while time.time() - start_time < time_limit and T > T_end:
        # Pick random circle
        i = np.random.randint(n_circles)
        
        # Propose move
        # 50% chance to move center, 50% to change radius
        if np.random.rand() < 0.5:
            # Move center
            step = T * 0.1
            dx = np.random.uniform(-step, step)
            dy = np.random.uniform(-step, step)
            new_x = x0[2*i] + dx
            new_y = x0[2*i+1] + dy
            new_r = x0[2*i+2]
        else:
            # Change radius
            step = T * 0.05
            new_r = x0[2*i+2] + np.random.uniform(-step, step)
            new_x = x0[2*i]
            new_y = x0[2*i+1]
            
        # Check bounds
        if new_x < 0 or new_x > w or new_y < 0 or new_y > h or new_r < 0:
            T *= cooling
            continue
            
        # Check wall constraints
        if new_r > new_x or new_r > w - new_x or new_r > new_y or new_r > h - new_y:
            T *= cooling
            continue
            
        # Check overlap
        valid = True
        for j in range(n_circles):
            if j == i:
                continue
            dx = new_x - x0[2*j]
            dy = new_y - x0[2*j+1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < new_r + x0[2*j+2]:
                valid = False
                break
                
        if not valid:
            T *= cooling
            continue
            
        # Calculate delta score
        old_r = x0[2*i+2]
        new_score = best_score + (new_r - old_r)
        delta = new_score - best_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta > 0 or np.random.rand() < np.exp(-delta / T):
            x0[2*i] = new_x
            x0[2*i+1] = new_y
            x0[2*i+2] = new_r
            best_score = new_score
            if best_score > np.sum(best_x[2::2]):
                best_x = x0.copy()
                
        T *= cooling
        
    return best_x, np.sum(best_x[2::2])

def _validate_solution(pos_r, w, h, n_circles, tol=1e-5):
    for i in range(n_circles):
        xi = pos_r[2*i]
        yi = pos_r[2*i+1]
        ri = pos_r[2*i+2]
        
        if xi < -tol or xi > w + tol or yi < -tol or yi > h + tol or ri < -tol:
            return False
        if ri - xi < -tol or ri - (w-xi) < -tol or ri - yi < -tol or ri - (h-yi) < -tol:
            return False
            
    for i in range(n_circles):
        for j in range(i+1, n_circles):
            xi = pos_r[2*i]
            yi = pos_r[2*i+1]
            ri = pos_r[2*i+2]
            xj = pos_r[2*j]
            yj = pos_r[2*j+1]
            rj = pos_r[2*j+2]
            
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < (ri + rj) - tol:
                return False
    return True

def _pack_for_aspect(w, h, num_starts=3, time_limit=40):
    n_circles = N
    best_x = None
    best_score = 0.0
    start_time = time.time()
    
    for s in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Add noise to grid init
        x0 = np.zeros(n_circles * 3)
        for i in range(n_circles):
            col = i % 5
            row = i // 5
            cx = (col + 0.5) * w / 5.0 + np.random.uniform(-0.05*w, 0.05*w)
            cy = (row + 0.5) * h / 5.0 + np.random.uniform(-0.05*h, 0.05*h)
            x0[2*i] = np.clip(cx, 0, w)
            x0[2*i+1] = np.clip(cy, 0, h)
            x0[2*i+2] = 0.01
            
        # Run SA
        x_res, score = _sa_pack(w, h, time_limit / num_starts)
        
        # Validate
        if _validate_solution(x_res, w, h, n_circles):
            if score > best_score:
                best_score = score
                best_x = x_res
        else:
            # Try to fix
            # (SA should produce valid, but just in case)
            pass

    if best_x is None:
        return None, 0.0
        
    positions = np.column_stack([best_x[0::2], best_x[1::2]])
    radii = best_x[2::2]
    radii = np.maximum(radii, 0)
    
    score = np.sum(radii)
    
    return np.column_stack([positions, radii]), score

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_result = None
    best_score = 0.0
    
    aspect_configs = [
        (1.0, 1.0),
        (1.2, 0.8),
        (0.8, 1.2),
        (1.4, 0.6),
        (0.6, 1.4),
        (1.5, 0.5),
        (1.1, 0.9),
        (0.9, 1.1)
    ]
    
    for w, h in aspect_configs:
        if time.time() > deadline:
            break
            
        time_remaining = deadline - time.time()
        if time_remaining < 10:
            continue
            
        num_starts = 3 if time_remaining > 60 else 1
        
        result, score = _pack_for_aspect(w, h, num_starts=num_starts, time_limit=time_remaining - 5)
        
        if result is not None and score > best_score:
            best_score = score
            best_result = result
            
    if best_result is None:
        w, h = 1.0, 1.0
        cols = 5
        rows = 5
        positions = []
        radii = []
        count = 0
        for i in range(cols):
            for j in range(rows):
                if count >= N:
                    break
                pos = [(i + 0.5) * w / cols, (j + 0.5) * h / rows]
                r = 0.5 * min(w/cols, h/rows) * 0.95
                positions.append(pos)
                radii.append(r)
                count += 1
        positions = np.array(positions)
        radii = np.array(radii)
        best_result = np.column_stack([positions, radii])
        
    return best_result
