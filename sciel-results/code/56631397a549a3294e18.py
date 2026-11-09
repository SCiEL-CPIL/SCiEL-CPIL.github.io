import numpy as np
import time
import sys

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 550.0  # Safety margin
    
    best_centers = None
    best_sides = None
    best_score = -1.0

    def check_validity(centers, sides, tol=1e-7):
        # Check bounds
        if np.any(sides <= 0) or np.any(~np.isfinite(sides)) or np.any(~np.isfinite(centers)):
            return False
        if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
            return False
        if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
            return False
        
        # Check overlaps
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

    # 2. Optimize Equal Squares
    def obj_equal(params):
        s = params[0]
        cx = params[1:1 + n]
        cy = params[1 + n:1 + 2 * n]
        
        # Bounds check penalty
        if s <= 0:
            return 1e6
        if np.any(cx < s/2) or np.any(cx > 1 - s/2) or np.any(cy < s/2) or np.any(cy > 1 - s/2):
            return 1e6
            
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t = s # since s_i = s_j = s, (s+s)/2 = s
                # Overlap if dx < t and dy < t
                if dx < t and dy < t:
                    # Penalty proportional to overlap depth
                    ov_x = t - dx
                    ov_y = t - dy
                    penalty += min(ov_x, ov_y) ** 2
        return -n * s + 100.0 * penalty

    # Initial guess for equal squares
    s_init = 0.275
    cx_init = np.array([(i % 4 + 0.5) * 0.25 for i in range(n)])
    cy_init = np.array([(i // 4 + 0.5) * 0.25 for i in range(n)])
    # Adjust cx, cy to be within [s/2, 1-s/2]
    cx_init = np.clip(cx_init, s_init/2, 1 - s_init/2)
    cy_init = np.clip(cy_init, s_init/2, 1 - s_init/2)
    
    x0_equal = np.concatenate([[s_init], cx_init, cy_init])
    bounds_equal = [(0.2, 0.35)] + [(0.1, 0.9)] * (2 * n)
    
    try:
        from scipy.optimize import differential_evolution
        result_equal = differential_evolution(
            obj_equal, 
            bounds_equal, 
            maxiter=200, 
            popsize=15, 
            seed=42,
            tol=1e-8,
            init='sobol'
        )
        if result_equal.success or result_equal.fun < 0: # Heuristic
            s_opt = result_equal.x[0]
            cx_opt = result_equal.x[1:1 + n]
            cy_opt = result_equal.x[1 + n:1 + 2 * n]
            centers_eq = np.stack([cx_opt, cy_opt], axis=1)
            sides_eq = np.full(n, s_opt)
            update_best(centers_eq, sides_eq)
    except Exception as e:
        pass

    if time.time() - start_time > time_limit * 0.5:
        if best_centers is not None:
            return best_centers, best_sides
        return centers_grid, sides_grid

    # 3. Optimize Free Squares
    def obj_free(params):
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

    # Initial guess: Grid
    s_init_free = np.full(n, 0.25)
    cx_init_free = np.array([(i % 4 + 0.5) * 0.25 for i in range(n)])
    cy_init_free = np.array([(i // 4 + 0.5) * 0.25 for i in range(n)])
    x0_free = np.concatenate([s_init_free, cx_init_free, cy_init_free])
    bounds_free = [(0.05, 0.6)] * n + [(0.05, 0.95)] * (2 * n)
    
    try:
        result_free = differential_evolution(
            obj_free, 
            bounds_free, 
            maxiter=300, 
            popsize=20, 
            seed=42,
            tol=1e-8,
            init='sobol'
        )
        if result_free.success or result_free.fun < 0:
            s_opt = result_free.x[0:n]
            cx_opt = result_free.x[n:2 * n]
            cy_opt = result_free.x[2 * n:3 * n]
            centers_fr = np.stack([cx_opt, cy_opt], axis=1)
            sides_fr = s_opt
            update_best(centers_fr, sides_fr)
    except Exception as e:
        pass

    if best_centers is None:
        return centers_grid, sides_grid

    return best_centers, best_sides
