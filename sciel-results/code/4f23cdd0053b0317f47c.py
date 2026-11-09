import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 550.0  # Safety margin

    best_centers = None
    best_sides = None
    best_score = -1.0

    def check_validity(centers, sides, tol=1e-7):
        if np.any(sides <= 0) or np.any(~np.isfinite(sides)) or np.any(~np.isfinite(centers)):
            return False
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        
        for i in range(n):
            for j in range(i + 1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    return False
        return True

    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if check_validity(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    def _unflatten(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def _penalty(x, tol=1e-7, weight=1e4):
        centers, sides = _unflatten(x)
        pen = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                pen += weight * (1.0 - s)
            if cx - s/2 < -tol:
                pen += weight * (-tol - (cx - s/2))
            if cx + s/2 > 1 + tol:
                pen += weight * ((cx + s/2) - (1 + tol))
            if cy - s/2 < -tol:
                pen += weight * (-tol - (cy - s/2))
            if cy + s/2 > 1 + tol:
                pen += weight * ((cy + s/2) - (1 + tol))
        
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if abs(dx) < t and abs(dy) < t:
                    pen += weight * ((t - abs(dx)) + (t - abs(dy)))
        return pen

    def _objective(x, tol=1e-7, weight=1e4):
        centers, sides = _unflatten(x)
        return -np.sum(sides) + _penalty(x, tol, weight)

    def _local_optimize(x0, tol=1e-7, time_limit=5):
        """Run Nelder-Mead or L-BFGS-B on flattened vector."""
        start = time.time()
        
        def obj(x):
            if time.time() - start > time_limit:
                return 1e10
            return _objective(x, tol)
        
        # Try L-BFGS-B with bounds first (often faster and more robust)
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # cx
            bounds.append((0.0, 1.0))  # cy
            bounds.append((0.01, 1.0)) # side
            
        try:
            result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if result.fun < 1e9:
                return result.x
        except Exception:
            pass
        
        # Fallback to Nelder-Mead
        try:
            result = minimize(obj, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < 1e9:
                return result.x
        except Exception:
            pass
            
        return x0

    # 1. Baseline Grid
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers_grid = np.stack([cx, cy], axis=1)
    sides_grid = np.full(n, side)
    update_best(centers_grid, sides_grid)

    # 2. Structured Seeds
    def make_seed_2big_8small():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s_big = 0.5
        centers[0] = [0.25, 0.75]
        centers[1] = [0.75, 0.75]
        sides[0] = s_big
        sides[1] = s_big
        
        s_small = 0.25
        for i in range(4):
            centers[2+i] = [(i+0.5)*s_small, 0.375]
            sides[2+i] = s_small
            centers[6+i] = [(i+0.5)*s_small, 0.125]
            sides[6+i] = s_small
        return centers, sides

    def make_seed_3big_7small():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        s_big = 1.0 / 3
        for i in range(3):
            centers[i] = [(i + 0.5) * s_big, 1.0 - s_big/2]
            sides[i] = s_big
            
        s_small = 0.2
        for i in range(4):
            centers[3+i] = [(i + 0.5) * s_small, 0.6]
            sides[3+i] = s_small
        for i in range(3):
            centers[7+i] = [(i + 0.5) * s_small + 0.1, 0.3]
            sides[7+i] = s_small
        return centers, sides

    seeds = []
    seeds.append((centers_grid, sides_grid))
    seeds.append(make_seed_2big_8small())
    seeds.append(make_seed_3big_7small())

    # 3. Global Search (Differential Evolution)
    # We run DE to find good candidates, then refine them.
    # To save time, we limit DE iterations and use the time budget wisely.
    
    def obj_de(params):
        s = params[0:n]
        cx = params[n:2 * n]
        cy = params[2 * n:3 * n]
        
        if np.any(s <= 0.01):
            return 1e6
        if np.any(cx < s/2) or np.any(cx > 1 - s/2) or np.any(cy < s/2) or np.any(cy > 1 - s/2):
            return 1e6
            
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = (s[i] + s[j]) / 2
                if dx < t and dy < t:
                    ov_x = t - dx
                    ov_y = t - dy
                    penalty += min(ov_x, ov_y) ** 2
        return -np.sum(s) + 100.0 * penalty

    bounds_de = [(0.05, 0.6)] * n + [(0.05, 0.95)] * (2 * n)
    
    # Run DE with a time check inside the objective if possible, or just limit iterations
    # Since DE is slow, we run it for a fixed number of iterations or until time is up.
    # We'll wrap it in a try-except and check time before starting.
    
    if time.time() - start_time < time_limit * 0.5:
        try:
            result_de = differential_evolution(
                obj_de, 
                bounds_de, 
                maxiter=100, 
                popsize=15, 
                seed=42,
                tol=1e-6,
                init='sobol'
            )
            if result_de.success or result_de.fun < 0:
                s_opt = result_de.x[0:n]
                cx_opt = result_de.x[n:2 * n]
                cy_opt = result_de.x[2 * n:3 * n]
                centers_de = np.stack([cx_opt, cy_opt], axis=1)
                sides_de = s_opt
                seeds.append((centers_de, sides_de))
        except Exception:
            pass

    # 4. Local Refinement of All Seeds
    # Sort seeds by initial score (best first) to prioritize refinement
    def seed_score(c, s):
        if not check_validity(c, s):
            return -1e6
        return np.sum(s)
    
    seeds.sort(key=lambda x: seed_score(x[0], x[1]), reverse=True)
    
    for c, s in seeds:
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Flatten
        x0 = np.concatenate([c.flatten(), s])
        
        # Refine
        x_opt = _local_optimize(x0, time_limit=10)
        
        # Unflatten and update
        c_opt, s_opt = _unflatten(x_opt)
        update_best(c_opt, s_opt)

    if best_centers is None:
        return centers_grid, sides_grid

    return best_centers, best_sides
