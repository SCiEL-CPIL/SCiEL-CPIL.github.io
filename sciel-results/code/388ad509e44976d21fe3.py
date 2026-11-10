# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Returns (centers, sides) where centers is (10, 2) and sides is (10,).
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin

    def _is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    return False
        return True

    def _max_side_for_square(cx_i, cy_i, s_i, other_cx, other_cy, other_s, i):
        """
        Compute maximum side for square i given its center and other squares.
        Returns the largest s such that square i doesn't overlap any other square
        and stays within the unit square.
        """
        # Boundary constraint
        max_s = min(2.0 * cx_i, 2.0 * (1.0 - cx_i), 2.0 * cy_i, 2.0 * (1.0 - cy_i))
        max_s = max(max_s, 1e-10)
        
        # Overlap constraints with other squares
        for j in range(10):
            if j == i:
                continue
            dx = abs(cx_i - other_cx[j])
            dy = abs(cy_i - other_cy[j])
            s_j = other_s[j]
            
            # For no overlap: need dx >= (s_i + s_j)/2 OR dy >= (s_i + s_j)/2
            # If dx >= (s_i + s_j)/2, then s_i <= 2*dx - s_j
            # If dy >= (s_i + s_j)/2, then s_i <= 2*dy - s_j
            # So s_i <= max(2*dx - s_j, 2*dy - s_j)
            # But we also need s_i > 0
            limit_x = 2.0 * dx - s_j
            limit_y = 2.0 * dy - s_j
            limit = max(limit_x, limit_y)
            if limit < max_s:
                max_s = limit
        
        return max(1e-10, max_s)

    def _compute_optimal_sides(centers):
        """
        Given centers, compute the maximum valid side for each square.
        Uses iterative refinement since sides affect each other.
        """
        cx = centers[:, 0].copy()
        cy = centers[:, 1].copy()
        s = np.full(10, 0.1)  # Initial guess
        
        for _ in range(50):
            s_new = np.zeros(10)
            for i in range(10):
                s_new[i] = _max_side_for_square(cx[i], cy[i], s[i], cx, cy, s, i)
            # Damped update
            s = 0.7 * s + 0.3 * s_new
            if np.max(np.abs(s - s_new)) < 1e-12:
                break
        
        # Final pass without damping
        for i in range(10):
            s[i] = _max_side_for_square(cx[i], cy[i], s[i], cx, cy, s, i)
        
        return np.maximum(s, 1e-10)

    def _objective_centers(x):
        """
        Objective: -sum of optimal sides given centers x = [cx0..cx9, cy0..cy9]
        """
        centers = np.column_stack([x[0:10], x[10:20]])
        s = _compute_optimal_sides(centers)
        return -np.sum(s)

    def _optimize_centers(centers, max_time=30.0, method='SLSQP'):
        """
        Optimize centers to maximize sum of optimal sides.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1]])
        bounds = [(0.0, 1.0)] * 20
        
        t0 = time.time()
        call_count = [0]
        
        def callback(xk):
            call_count[0] += 1
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        try:
            if method == 'SLSQP':
                res = minimize(_objective_centers, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 300, 'ftol': 1e-12, 'maxfun': 2000},
                               callback=callback)
            else:
                res = minimize(_objective_centers, x0, method='COBYLA',
                               options={'maxiter': 300, 'rhobeg': 0.05},
                               callback=callback)
            
            if res.success or (time.time() - t0 < max_time):
                x_opt = res.x
                new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
                new_centers = np.clip(new_centers, 0.0, 1.0)
                new_sides = _compute_optimal_sides(new_centers)
                
                if _is_valid(new_centers, new_sides):
                    return new_centers, new_sides
        except Exception:
            pass
        
        return None

    def _optimize_full(centers, sides, max_time=30.0, method='SLSQP'):
        """
        Full 30-dim optimization with penalty for overlaps.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1], sides])
        bounds = [(0.0, 1.0)] * 20 + [(0.01, 1.0)] * 10
        
        t0 = time.time()
        
        def _obj(x, penalty_weight=500.0):
            cx = x[0:10]
            cy = x[10:20]
            s = x[20:30]
            
            obj = -np.sum(s)
            
            # Boundary penalties
            for i in range(10):
                v = cx[i] - s[i]/2.0
                if v < 0:
                    obj += penalty_weight * v * v
                v = (1.0 - s[i]/2.0) - cx[i]
                if v < 0:
                    obj += penalty_weight * v * v
                v = cy[i] - s[i]/2.0
                if v < 0:
                    obj += penalty_weight * v * v
                v = (1.0 - s[i]/2.0) - cy[i]
                if v < 0:
                    obj += penalty_weight * v * v
            
            # Overlap penalties
            eps = 1e-8
            for i in range(10):
                for j in range(i+1, 10):
                    dx = abs(cx[i] - cx[j])
                    dy = abs(cy[i] - cy[j])
                    t = (s[i] + s[j]) / 2.0
                    ov_x = t - dx
                    ov_y = t - dy
                    smax_u = 0.5 * (ov_x + np.sqrt(ov_x**2 + eps))
                    smax_v = 0.5 * (ov_y + np.sqrt(ov_y**2 + eps))
                    obj += penalty_weight * smax_u * smax_v
            
            return obj
        
        def callback(xk):
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
        
        try:
            if method == 'SLSQP':
                res = minimize(_obj, x0, method='SLSQP', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-10, 'maxfun': 1500},
                               callback=callback)
            else:
                res = minimize(_obj, x0, method='COBYLA',
                               options={'maxiter': 200, 'rhobeg': 0.05},
                               callback=callback)
            
            if res.success or (time.time() - t0 < max_time):
                x_opt = res.x
                new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
                new_sides = x_opt[20:30]
                
                # Repair
                new_centers, new_sides = _repair_and_clamp(new_centers, new_sides)
                
                if _is_valid(new_centers, new_sides):
                    return new_centers, new_sides
        except Exception:
            pass
        
        return None

    def _repair_and_clamp(centers, sides):
        """
        Ensure validity by shrinking sides if necessary.
        """
        centers = centers.copy()
        sides = sides.copy()
        
        for i in range(10):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
        
        for _ in range(300):
            overlap_found = False
            for i in range(10):
                for j in range(i+1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        overlap_found = True
                        if sides[i] < sides[j]:
                            sides[i] *= 0.85
                        else:
                            sides[j] *= 0.85
                        idx = i if sides[i] < sides[j] else j
                        h = sides[idx] / 2.0
                        centers[idx, 0] = np.clip(centers[idx, 0], h, 1.0 - h)
                        centers[idx, 1] = np.clip(centers[idx, 1], h, 1.0 - h)
            if not overlap_found:
                break
        
        return centers, sides

    def _update_best(c, s, best_holder):
        if _is_valid(c, s):
            score = np.sum(s)
            if score > best_holder[0]:
                best_holder[0] = score
                best_holder[1] = c.copy()
                best_holder[2] = s.copy()

    best_score = -1.0
    best_centers = None
    best_sides = None
    best_holder = [best_score, best_centers, best_sides]

    # --- Initial configurations ---
    
    def init_grid_33():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.15, 0.5, 0.85]
        idx = 0
        for r in range(3):
            for col in range(3):
                c[idx, 0] = coords[col]
                c[idx, 1] = coords[r]
                s[idx] = 0.3
                idx += 1
        c[9, 0] = 0.5
        c[9, 1] = 0.5
        s[9] = 0.05
        return c, s

    def init_corner():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        c[8] = [0.35, 0.35]; s[8] = 0.1
        c[9] = [0.65, 0.65]; s[9] = 0.1
        return c, s

    def init_stripe():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.18
        return c, s

    def init_diagonal():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            t = (i + 0.5) / 10.0
            c[i, 0] = t
            c[i, 1] = t
            s[i] = 0.25
        return c, s

    def init_random(seed):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.1, 0.9, size=(10, 2))
        s = rng.uniform(0.1, 0.4, size=10)
        return c, s

    # 4x4 grid with one removed
    def init_grid_44():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        coords = [0.125, 0.375, 0.625, 0.875]
        idx = 0
        for r in range(4):
            for col in range(4):
                if idx < 10:
                    c[idx, 0] = coords[col]
                    c[idx, 1] = coords[r]
                    s[idx] = 0.24
                    idx += 1
        return c, s

    # 2 rows of 5
    def init_2x5():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            row = i // 5
            col = i % 5
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.19
        return c, s

    # 5 columns of 2
    def init_5x2():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        for i in range(10):
            col = i // 2
            row = i % 2
            c[i, 0] = 0.1 + col * 0.2
            c[i, 1] = 0.25 + row * 0.5
            s[i] = 0.19
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        init_stripe,
        init_diagonal,
        init_grid_44,
        init_2x5,
        init_5x2,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999),
        lambda: init_random(777),
        lambda: init_random(2024),
        lambda: init_random(555),
        lambda: init_random(314),
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
        
        c, s = init_fn()
        c, s = _repair_and_clamp(c, s)
        _update_best(c, s, best_holder)
        
        # Optimize centers (20-dim) - this is the key improvement
        result = _optimize_centers(c, max_time=35.0, method='SLSQP')
        if result is not None:
            _update_best(result[0], result[1], best_holder)
        
        if time.time() > deadline:
            break
        
        # Second pass: COBYLA on centers
        if result is not None:
            c2, s2 = result
        else:
            c2, s2 = c, s
        
        result2 = _optimize_centers(c2, max_time=35.0, method='COBYLA')
        if result2 is not None:
            _update_best(result2[0], result2[1], best_holder)
        
        # Full 30-dim optimization from best
        if best_holder[1] is not None and time.time() < deadline - 80:
            result3 = _optimize_full(best_holder[1], best_holder[2], max_time=30.0, method='SLSQP')
            if result3 is not None:
                _update_best(result3[0], result3[1], best_holder)
        
        # Perturbation-based local search
        if best_holder[1] is not None and time.time() < deadline - 60:
            rng = np.random.default_rng(i * 100 + 1)
            c3 = best_holder[1] + rng.normal(0, 0.04, size=(10, 2))
            c3 = np.clip(c3, 0.0, 1.0)
            s3 = _compute_optimal_sides(c3)
            _update_best(c3, s3, best_holder)
            
            result4 = _optimize_centers(c3, max_time=25.0, method='SLSQP')
            if result4 is not None:
                _update_best(result4[0], result4[1], best_holder)

    # Final polish
    if best_holder[1] is not None and time.time() < deadline - 30:
        result_final = _optimize_centers(best_holder[1], max_time=25.0, method='SLSQP')
        if result_final is not None:
            _update_best(result_final[0], result_final[1], best_holder)
        
        # One more COBYLA polish
        if time.time() < deadline - 15:
            result_final2 = _optimize_centers(best_holder[1], max_time=15.0, method='COBYLA')
            if result_final2 is not None:
                _update_best(result_final2[0], result_final2[1], best_holder)

    if best_holder[1] is None:
        best_centers = np.zeros((10, 2))
        best_sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            col = i % 5
            best_centers[i, 0] = 0.1 + col * 0.2
            best_centers[i, 1] = 0.1 + r * 0.2
            best_sides[i] = 0.1
        best_centers, best_sides = _repair_and_clamp(best_centers, best_sides)
        best_score = np.sum(best_sides)
    else:
        best_score = best_holder[0]
        best_centers = best_holder[1]
        best_sides = best_holder[2]

    return best_centers, best_sides

# EVOLVE-BLOCK-END
