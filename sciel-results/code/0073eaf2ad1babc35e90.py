"""Circle packing in perimeter-4 rectangle: maximize sum of radii for 21 circles."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21
DEADLINE = 140  # seconds, safety margin before 150s limit

def construct():
    start_time = time.time()
    best_score = -1.0
    best_result = None

    def _check_validity(circles, w, h):
        """Check if circles are valid (non-overlapping, within bounds)."""
        if len(circles) != N:
            return False
        for i in range(N):
            x, y, r = circles[i]
            if r < 0:
                return False
            if x - r < -1e-9 or x + r > w + 1e-9:
                return False
            if y - r < -1e-9 or y + r > h + 1e-9:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < ri + rj - 1e-9:
                    return False
        return True

    def _score(circles):
        if _check_validity(circles, best_result_w, best_result_h):
            return float(np.sum(circles[:, 2]))
        return -1.0

    # We need to track w and h for the best result
    best_result_w = 1.0
    best_result_h = 1.0

    def _update_best(circles, w, h):
        nonlocal best_score, best_result, best_result_w, best_result_h
        if _check_validity(circles, w, h):
            score = float(np.sum(circles[:, 2]))
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_result_w = w
                best_result_h = h

    def _grid_packing(rows, cols, w, h):
        """
        Compute a grid-based packing for rows x cols grid in rectangle w x h.
        Returns (positions, radii) or (None, None) if invalid.
        """
        if rows * cols < N:
            return None, None

        positions = np.zeros((N, 2))

        for i in range(N):
            row = i // cols
            col = i % cols

            if row == rows - 1:
                count_in_last_row = N - (rows - 1) * cols
                if count_in_last_row < cols:
                    k = count_in_last_row
                    x = (col + 0.5) * w / k
                else:
                    x = (col + 0.5) * w / cols
            else:
                x = (col + 0.5) * w / cols

            y = (row + 0.5) * h / rows
            positions[i] = [x, y]

        # Compute maximum valid radius
        min_r = float('inf')

        # Boundary constraints
        for i in range(N):
            min_r = min(min_r, positions[i, 0], w - positions[i, 0],
                       positions[i, 1], h - positions[i, 1])

        # Pairwise distance constraints
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_r = min(min_r, dist / 2.0)

        if min_r <= 0:
            return None, None

        radii = np.full(N, min_r)
        return positions, radii

    def _objective(v, w, h):
        """
        Objective function for optimization.
        v: [x0, y0, r0, x1, y1, r1, ...]
        Minimize penalty for violations and maximize sum of radii.
        """
        xs = v[0:3*N:3]
        ys = v[1:3*N:3]
        rs = v[2:3*N:3]
        
        # Ensure positive radii
        rs = np.maximum(rs, 0.0)
        
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            # x boundaries
            if xs[i] - rs[i] < 0:
                penalty += 100.0 * (xs[i] - rs[i])**2
            if xs[i] + rs[i] > w:
                penalty += 100.0 * (xs[i] + rs[i] - w)**2
            # y boundaries
            if ys[i] - rs[i] < 0:
                penalty += 100.0 * (ys[i] - rs[i])**2
            if ys[i] + rs[i] > h:
                penalty += 100.0 * (ys[i] + rs[i] - h)**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += 100.0 * (min_dist - dist)**2
        
        # We want to maximize sum(rs), so we minimize -sum(rs)
        return penalty - np.sum(rs)

    def _run_optimization(w_init, x0, y0, r0, max_iter=500):
        """Run Powell optimization for given initial conditions."""
        h = 2.0 - w_init
        if h <= 0.01 or w_init <= 0.01:
            return None, -1.0

        v0 = np.zeros(3 * N)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]

        bounds = []
        for i in range(N):
            bounds.append((0, w_init))  # x
            bounds.append((0, h))       # y
            bounds.append((0, 1.0))     # r

        try:
            # Powell is derivative-free and robust for non-convex problems
            res = minimize(_objective, v0, args=(w_init, h), method='Powell',
                          bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if res.fun < _objective(v0, w_init, h):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                circles = np.column_stack([xs, ys, rs])
                _update_best(circles, w_init, h)
                return circles, _score(circles)
        except Exception:
            pass

        return None, -1.0

    def _local_radius_expansion(circles, w, h, steps=50):
        """
        Try to increase radii by checking local feasibility.
        This is a greedy local search.
        """
        circles = circles.copy()
        improved = True
        while improved and steps > 0:
            improved = False
            steps -= 1
            for i in range(N):
                current_r = circles[i, 2]
                # Try to increase radius
                new_r = current_r * 1.01
                # Check if it fits in bounds
                x, y = circles[i, 0], circles[i, 1]
                if x - new_r < 0 or x + new_r > w or y - new_r < 0 or y + new_r > h:
                    continue
                
                # Check if it overlaps with others
                valid = True
                for j in range(N):
                    if i == j: continue
                    dx = x - circles[j, 0]
                    dy = y - circles[j, 1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < new_r + circles[j, 2] - 1e-9:
                        valid = False
                        break
                
                if valid:
                    circles[i, 2] = new_r
                    improved = True
    
    return best_result

    # Phase 1: Grid-based packings with various aspect ratios
    candidate_configs = []
    for rows in range(1, 10):
        for cols in range(1, 10):
            if rows * cols >= N:
                ar = cols / rows
                candidate_configs.append((rows, cols, ar))

    # Add non-integer aspect ratios
    for ar in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0]:
        best_rows, best_cols = None, None
        best_diff = float('inf')
        for rows in range(1, 10):
            for cols in range(1, 10):
                if rows * cols >= N:
                    grid_ar = cols / rows
                    diff = abs(grid_ar - ar)
                    if diff < best_diff:
                        best_diff = diff
                        best_rows, best_cols = rows, cols
        if best_rows is not None:
            candidate_configs.append((best_rows, best_cols, ar))

    # Remove duplicates
    seen = set()
    unique_configs = []
    for rows, cols, ar in candidate_configs:
        key = (rows, cols, round(ar, 3))
        if key not in seen:
            seen.add(key)
            unique_configs.append((rows, cols, ar))

    # Sort configs by "squareness" (closer to 1.0 aspect ratio is often better for packing)
    unique_configs.sort(key=lambda x: abs(x[2] - 1.0))

    for rows, cols, ar in unique_configs:
        if time.time() - start_time > DEADLINE - 20:
            break

        h = 2.0 / (ar + 1.0)
        w = ar * h

        positions, radii = _grid_packing(rows, cols, w, h)
        if positions is not None:
            circles = np.column_stack([positions, radii])
            _update_best(circles, w, h)

    # Phase 2: Optimization from grid initializations
    # We limit the number of optimizations to fit in time
    max_opts = 10
    opt_count = 0
    for rows, cols, ar in unique_configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        if opt_count >= max_opts:
            break

        h = 2.0 / (ar + 1.0)
        w = ar * h

        positions, radii = _grid_packing(rows, cols, w, h)
        if positions is not None:
            # Add slight jitter to break symmetry
            np.random.seed(42 + rows + cols)
            jitter = 0.01 * np.random.randn(N, 2)
            x0 = positions[:, 0] + jitter[:, 0]
            y0 = positions[:, 1] + jitter[:, 1]
            r0 = radii.copy()
            
            # Ensure initial state is roughly valid for optimizer
            # Clamp positions to valid range
            x0 = np.clip(x0, 0, w)
            y0 = np.clip(y0, 0, h)
            
            _run_optimization(w, x0, y0, r0, max_iter=200)
            opt_count += 1

    # Phase 3: Local radius expansion on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 2:
        # Try to expand radii locally
        expanded = _local_radius_expansion(best_result, best_result_w, best_result_h, steps=100)
        _update_best(expanded, best_result_w, best_result_h)

    # Fallback: return a valid grid configuration if nothing better found
    if best_result is None:
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
        best_result = np.array(cells)
        best_score = _score(best_result)
        best_result_w = 1.0
        best_result_h = 1.0

    return best_result
