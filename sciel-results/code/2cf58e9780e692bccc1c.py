"""Circle packing in perimeter-4 rectangle: force-directed multi-start optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# Deadline guard
_DEADLINE = 120.0  # seconds
_START_TIME = None


def _elapsed():
    return time.time() - _START_TIME


def _perimeter_constraint(W, H):
    """Perimeter = 2*(W+H) <= 4, so W+H <= 2."""
    return 2.0 - (W + H)


def _pack_single(aspect, n_starts=3, max_iter=5000, tol=1e-10):
    """
    For a given aspect ratio (W/H), find W, H with W+H=2, and pack N circles.
    Maximize sum of radii.
    
    Variables: x_i, y_i (positions), r_i (radii), and we fix W,H from aspect.
    Constraints:
      - x_i >= r_i, x_i <= W - r_i
      - y_i >= r_i, y_i <= H - r_i
      - (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    """
    W = 2.0 * aspect / (1.0 + aspect)
    H = 2.0 / (1.0 + aspect)
    
    best_sum = 0.0
    best_solution = None
    
    for start in range(n_starts):
        # Initial configuration: grid-like with some randomness
        if start == 0:
            # Grid initialization
            cols = int(np.ceil(np.sqrt(N * W / H)))
            rows = int(np.ceil(N / cols))
            xs = np.zeros(N)
            ys = np.zeros(N)
            rs = np.zeros(N)
            for i in range(N):
                c = i % cols
                r = i // cols
                xs[i] = (c + 0.5) * W / cols
                ys[i] = (r + 0.5) * H / rows
                rs[i] = min(W / cols, H / rows) * 0.45
        else:
            # Random initialization
            rng = np.random.RandomState(start * 12345 + 678)
            margin = 0.05
            xs = rng.uniform(margin, W - margin, N)
            ys = rng.uniform(margin, H - margin, N)
            rs = rng.uniform(0.05, min(W, H) * 0.15, N)
        
        # Combine into single vector: [x0, y0, r0, x1, y1, r1, ...]
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3*i] = xs[i]
            x0[3*i+1] = ys[i]
            x0[3*i+2] = rs[i]
        
        def objective(v):
            """Negative sum of radii (we minimize)."""
            return -np.sum(v[2::3])
        
        def constraints(v):
            cons = []
            xs_v = v[0::3]
            ys_v = v[1::3]
            rs_v = v[2::3]
            
            # Boundary constraints: x >= r, x <= W-r, y >= r, y <= H-r
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: W - v[3*i] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: H - v[3*i+1] - v[3*i+2]})
            
            # Pairwise distance constraints: dist >= r_i + r_j
            for i in range(N):
                for j in range(i+1, N):
                    cons.append({
                        'type': 'ineq',
                        'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                    })
            
            # Positive radii
            for i in range(N):
                cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
            
            return cons
        
        # Use SLSQP
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=constraints,
                          options={'maxiter': max_iter, 'ftol': tol})
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum:
                    best_sum = sum_r
                    best_solution = res.x
        except Exception:
            pass
    
    if best_solution is None:
        return 0.0, None
    
    # Extract final positions and radii
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_solution[3*i]
        result[i, 1] = best_solution[3*i+1]
        result[i, 2] = best_solution[3*i+2]
    
    return best_sum, result


def _validate_and_fix(result, W, H):
    """Check validity and slightly shrink radii if needed to ensure no overlap."""
    if result is None:
        return None
    
    # Check all constraints and shrink if violated
    max_shrink = 1.0
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = result[i]
            if x - r < -1e-9 or x + r > W + 1e-9 or y - r < -1e-9 or y + r > H + 1e-9:
                valid = False
                break
        if not valid:
            max_shrink *= 0.95
            continue
        for i in range(N):
            for j in range(i+1, N):
                dx = result[i,0] - result[j,0]
                dy = result[i,1] - result[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < result[i,2] + result[j,2] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        max_shrink *= 0.95
    
    if max_shrink < 0.9:
        # Too much shrinking needed, might be invalid
        return None
    
    result[:, 2] *= max_shrink * (1 - 1e-6)
    return result


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 1.3, 1.1, 1.4, 0.9, 1.6, 1.25]
    
    best_sum = 0.0
    best_result = None
    best_W = 2.0
    best_H = 2.0
    
    for aspect in aspect_ratios:
        if _elapsed() > _DEADLINE - 15:
            break
        
        W = 2.0 * aspect / (1.0 + aspect)
        H = 2.0 / (1.0 + aspect)
        
        # Number of starts depends on remaining time
        remaining = _DEADLINE - 15 - _elapsed()
        if remaining < 10:
            n_starts = 1
        elif remaining < 30:
            n_starts = 2
        else:
            n_starts = 3
        
        sum_r, result = _pack_single(aspect, n_starts=n_starts, max_iter=3000, tol=1e-10)
        
        if result is not None:
            result = _validate_and_fix(result, W, H)
            if result is not None:
                actual_sum = np.sum(result[:, 2])
                if actual_sum > best_sum:
                    best_sum = actual_sum
                    best_result = result.copy()
                    best_W = W
                    best_H = H
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        W = 2.0
        H = 2.0
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * W / side
                y = (j + 0.5) * H / side
                r = min(W, H) / side * 0.45
                cells.append([x, y, r])
        return np.array(cells)
    
    return best_result
