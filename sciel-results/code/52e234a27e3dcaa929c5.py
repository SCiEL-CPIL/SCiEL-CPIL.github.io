"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 145.0  # safety margin before 150s kill

def construct():
    t0 = time.time()
    best_value = 0.0
    best_result = None
    
    # We'll try multiple optimization runs
    # Variables: [w, x1, y1, r1, x2, y2, r2, ..., x21, y21, r21]
    # w = width, h = 2 - w
    # n_vars = 1 + 3*N
    
    def make_objective(penalty):
        def objective(v):
            w = v[0]
            h = 2.0 - w
            if w <= 0.01 or h <= 0.01:
                return 1e10
            coords = v[1:].reshape(N, 3)  # x, y, r
            x = coords[:, 0]
            y = coords[:, 1]
            r = coords[:, 2]
            
            # Sum of radii (to maximize, so negate)
            obj = -np.sum(r)
            
            # Boundary violations
            for i in range(N):
                # Left
                viol = max(0.0, r[i] - x[i])
                obj += penalty * viol * viol
                # Right
                viol = max(0.0, x[i] + r[i] - w)
                obj += penalty * viol * viol
                # Bottom
                viol = max(0.0, r[i] - y[i])
                obj += penalty * viol * viol
                # Top
                viol = max(0.0, y[i] + r[i] - h)
                obj += penalty * viol * viol
            
            # Overlap violations
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    viol = max(0.0, r[i] + r[j] - dist)
                    obj += penalty * viol * viol
            
            return obj
        return objective
    
    def initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Try different aspect ratios
        w = 0.9 + 0.2 * rng.random()  # between 0.9 and 1.1
        h = 2.0 - w
        
        # Grid-based initial placement
        # Try a 5x5 grid with some randomness
        cols = 5
        rows = 5
        cell_w = w / cols
        cell_h = h / rows
        
        v = np.zeros(1 + 3 * N)
        v[0] = w
        
        idx = 0
        for i in range(cols):
            for j in range(rows):
                if idx >= N:
                    break
                cx = (i + 0.5) * cell_w + rng.uniform(-0.1, 0.1) * cell_w
                cy = (j + 0.5) * cell_h + rng.uniform(-0.1, 0.1) * cell_h
                r = 0.35 * min(cell_w, cell_h)
                v[1 + 3 * idx] = cx
                v[2 + 3 * idx] = cy
                v[3 + 3 * idx] = r
                idx += 1
        
        return v
    
    def check_validity(v):
        w = v[0]
        h = 2.0 - w
        if w <= 0.01 or h <= 0.01:
            return False
        coords = v[1:].reshape(N, 3)
        x = coords[:, 0]
        y = coords[:, 1]
        r = coords[:, 2]
        
        if np.any(r < 0):
            return False
        
        # Check boundaries
        for i in range(N):
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > w + 1e-6:
                return False
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > h + 1e-6:
                return False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-6:
                    return False
        
        return True
    
    def compute_value(v):
        if not check_validity(v):
            return -1.0
        coords = v[1:].reshape(N, 3)
        return np.sum(coords[:, 2])
    
    # Try multiple seeds with increasing penalty
    seeds = [42, 123, 7, 999, 555, 2024, 31415, 27182]
    
    for seed in seeds:
        if time.time() - t0 > DEADLINE - 30:
            break
        
        v0 = initial_guess(seed)
        
        # Start with moderate penalty, increase if needed
        penalties = [50.0, 200.0, 500.0]
        
        current_v = v0.copy()
        for pen in penalties:
            if time.time() - t0 > DEADLINE - 25:
                break
            
            try:
                res = minimize(
                    make_objective(pen),
                    current_v,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.success or res.fun < compute_value(current_v) if check_validity(current_v) else True:
                    current_v = res.x
            except Exception:
                pass
        
        # Polish: try to improve with higher penalty
        if time.time() - t0 > DEADLINE - 20:
            break
        
        try:
            res = minimize(
                make_objective(1000.0),
                current_v,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res.fun < 1e6:  # reasonable
                current_v = res.x
        except Exception:
            pass
        
        val = compute_value(current_v)
        if val > best_value:
            best_value = val
            best_result = current_v.copy()
    
    # If we didn't find anything valid, return a basic grid
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
        # Make it fit in a 1x1 square (perimeter 4)
        return np.array(cells[:N])
    
    # Extract result: [w, x1, y1, r1, ...] -> array of [x, y, r]
    coords = best_result[1:].reshape(N, 3)
    return coords
