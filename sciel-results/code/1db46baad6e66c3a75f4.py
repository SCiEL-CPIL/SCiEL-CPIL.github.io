"""Circle packing in perimeter-4 rectangle: combined grid + SLSQP optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_result = None
    best_score = -1.0

    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        if len(positions) != N:
            return
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            return
        if np.any(radii < -1e-9):
            return
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-9:
                return
            if positions[i, 0] + radii[i] > w + 1e-9:
                return
            if positions[i, 1] - radii[i] < -1e-9:
                return
            if positions[i, 1] + radii[i] > h + 1e-9:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def compute_optimal_grid_packing(rows, cols, w, h):
        """
        For a given grid of rows x cols, place N circles in row-major order.
        Returns (positions, radii) or (None, None).
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

    def run_optimization(w_init, x0, y0, r0, max_iter=300, ftol=1e-12):
        """Run SLSQP optimization for given initial conditions."""
        h = 2.0 - w_init
        if h <= 0.05 or w_init <= 0.05:
            return None, -1.0

        n_var = 3 * N + 1

        def objective(v):
            xs = v[0:3*N:3]
            ys = v[1:3*N:3]
            rs = v[2:3*N:3]
            w = v[3*N]
            h = 2.0 - w
            if h <= 0.05 or w <= 0.05:
                return 1e10
            # Penalty for out-of-bounds
            penalty = 0.0
            for i in range(N):
                if xs[i] - rs[i] < -1e-6:
                    penalty += 1000 * (xs[i] - rs[i] + 1e-6)**2
                if xs[i] + rs[i] > w + 1e-6:
                    penalty += 1000 * (xs[i] + rs[i] - w - 1e-6)**2
                if ys[i] - rs[i] < -1e-6:
                    penalty += 1000 * (ys[i] - rs[i] + 1e-6)**2
                if ys[i] + rs[i] > h + 1e-6:
                    penalty += 1000 * (ys[i] + rs[i] - h - 1e-6)**2
            # Non-overlap penalty
            for i in range(N):
                for j in range(i + 1, N):
                    dist = math.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                    min_d = rs[i] + rs[j]
                    if dist < min_d:
                        penalty += 1000 * (min_d - dist)**2
            return -np.sum(rs) + penalty

        bounds = []
        for i in range(N):
            bounds.append((0, 2.0))  # x
            bounds.append((0, 2.0))  # y
            bounds.append((0, 1.0))  # r
        bounds.append((0.3, 1.7))  # w

        v0 = np.zeros(n_var)
        for i in range(N):
            v0[i*3] = x0[i]
            v0[i*3+1] = y0[i]
            v0[i*3+2] = r0[i]
        v0[3*N] = w_init

        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': ftol})
            if res.success or res.fun < objective(v0):
                xs = res.x[0:3*N:3]
                ys = res.x[1:3*N:3]
                rs = res.x[2:3*N:3]
                w = res.x[3*N]
                h = 2.0 - w
                circles = np.column_stack([xs, ys, rs])
                validate_and_update(xs, rs, w, h)
                return circles, float(np.sum(rs))
        except Exception:
            pass

        return None, -1.0

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

    # Track best grid configs for optimization
    best_grid_configs = []

    for rows, cols, ar in unique_configs:
        if time.time() > deadline - 40:
            break

        h = 2.0 / (ar + 1.0)
        w = ar * h

        positions, radii = compute_optimal_grid_packing(rows, cols, w, h)
        if positions is not None:
            validate_and_update(positions, radii, w, h)
            # Track top grid configs for optimization
            score = float(np.sum(radii))
            best_grid_configs.append((score, positions.copy(), radii.copy(), w, h, rows, cols))

    # Sort grid configs by score and keep top ones for optimization
    best_grid_configs.sort(key=lambda x: -x[0])
    top_configs = best_grid_configs[:5]

    # Phase 2: Optimization from best grid initializations
    for score, positions, radii, w, h, rows, cols in top_configs:
        if time.time() > deadline - 15:
            break

        # Estimate w from bounding box
        w_est = max(np.max(positions[:, 0] + radii), 0.5)
        h_est = max(np.max(positions[:, 1] + radii), 0.5)
        scale = 2.0 / (w_est + h_est)
        w_polish = w_est * scale
        h_polish = h_est * scale

        if 0.3 < w_polish < 1.7:
            x0 = positions[:, 0] * (w_polish / w)
            y0 = positions[:, 1] * (h_polish / h)
            r0 = radii * min(w_polish / w, h_polish / h)
            run_optimization(w_polish, x0, y0, r0)

    # Phase 3: Try a few more diverse starting points
    if time.time() < deadline - 15:
        # Try square-ish and elongated configurations
        for ar in [0.7, 1.0, 1.4, 2.0]:
            if time.time() > deadline - 10:
                break
            h = 2.0 / (ar + 1.0)
            w = ar * h

            # Use 4x5 grid as base
            rows, cols = 4, 5
            positions, radii = compute_optimal_grid_packing(rows, cols, w, h)
            if positions is not None:
                score = float(np.sum(radii))
                if score > best_score - 0.01:
                    x0 = positions[:, 0]
                    y0 = positions[:, 1]
                    r0 = radii
                    run_optimization(w, x0, y0, r0)

    # Fallback
    if best_result is None:
        # Square packing in 1x1 rectangle
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)

    return best_result
