"""Circle packing in perimeter-4 rectangle using grid-cell exclusion + force relaxation + NLP polish."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 135.0  # Safety margin for 150s limit

    best_score = 0.0
    best_solution = None
    best_w = 1.0
    best_h = 1.0

    def evaluate_solution(x, w, h):
        """Validate solution and compute score."""
        for i in range(N):
            cx, cy, r = x[i]
            if cx - r < -1e-8 or cx + r > w + 1e-8:
                return -np.inf, False
            if cy - r < -1e-8 or cy + r > h + 1e-8:
                return -np.inf, False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = x[i, 2] + x[j, 2]
                if dist < min_dist - 1e-8:
                    return -np.inf, False
        return np.sum(x[:, 2]), True

    def force_relax(x, w, h, max_iter=300):
        """Force-directed relaxation to resolve overlaps and boundary violations."""
        x = x.copy().astype(float)
        dt = 0.005
        min_dt = 0.0001

        for it in range(max_iter):
            if time.time() - start_time > timeout:
                break

            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall repulsion
            for i in range(N):
                cx, cy, r = x[i]
                if cx - r < 0:
                    fx[i] += (cx - r) * 100
                if cx + r > w:
                    fx[i] -= (cx + r - w) * 100
                if cy - r < 0:
                    fy[i] += (cy - r) * 100
                if cy + r > h:
                    fy[i] -= (cy + r - h) * 100

            # Pairwise repulsion
            max_overlap = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i, 0] - x[j, 0]
                    dy = x[i, 1] - x[j, 1]
                    d_sq = dx * dx + dy * dy
                    min_d = x[i, 2] + x[j, 2]
                    if d_sq < min_d * min_d and d_sq > 1e-12:
                        d = np.sqrt(d_sq)
                        overlap = min_d - d
                        max_overlap = max(max_overlap, overlap)
                        f = overlap * 15
                        fx[i] += (dx / d) * f
                        fy[i] += (dy / d) * f
                        fx[j] -= (dx / d) * f
                        fy[j] -= (dy / d) * f

            if max_overlap < 1e-8:
                break

            force_mag = np.max(np.sqrt(fx * fx + fy * fy))
            if force_mag > 1.0:
                dt = max(min_dt, dt * 0.5)
            elif force_mag < 0.1:
                dt = min(0.01, dt * 1.1)

            x[:, 0] += dt * fx
            x[:, 1] += dt * fy

            # Clamp positions
            for i in range(N):
                x[i, 0] = np.clip(x[i, 0], x[i, 2], w - x[i, 2])
                x[i, 1] = np.clip(x[i, 1], x[i, 2], h - x[i, 2])

        return x

    def nlp_polish(x, w, h, max_iter=200):
        """NLP polish using scipy minimize with penalty-based objective."""
        x = x.copy().astype(float)

        def objective(params):
            p = params.reshape(N, 3)
            cx, cy, r = p[:, 0], p[:, 1], p[:, 2]

            pen = 0.0
            viol = np.clip(r - cx, 0, None)
            pen += np.sum(viol ** 2)
            viol = np.clip(cx + r - w, 0, None)
            pen += np.sum(viol ** 2)
            viol = np.clip(r - cy, 0, None)
            pen += np.sum(viol ** 2)
            viol = np.clip(cy + r - h, 0, None)
            pen += np.sum(viol ** 2)

            for i in range(N):
                for j in range(i + 1, N):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    viol = np.clip(min_d - dist, 0, None)
                    pen += viol ** 2

            return -np.sum(r) + 100.0 * pen

        x0 = x.flatten()
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w))
            bounds.append((0, h))
            bounds.append((0.001, max_r))

        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': max_iter, 'maxfun': 2000})
            return res.x.reshape(N, 3)
        except Exception:
            return x

    # Grid-cell exclusion method: assign circles to grid cells
    def grid_cell_placement(w, h, nx, ny):
        """Place circles using grid cell exclusion. Each circle is centered in its cell
        and its radius is bounded by the distance to cell boundaries and neighboring circles."""
        x = np.zeros((N, 3))
        cell_w = w / nx
        cell_h = h / ny

        # Create grid positions
        positions = []
        for i in range(nx):
            for j in range(ny):
                positions.append((i, j))

        # Shuffle to get diverse assignments
        rng = random.Random(42)
        rng.shuffle(positions)

        for idx in range(N):
            if idx >= len(positions):
                break
            i, j = positions[idx]
            cx = (i + 0.5) * cell_w
            cy = (j + 0.5) * cell_h

            # Radius bounded by distance to cell boundaries
            r_max = min(cx, w - cx, cy, h - cy) * 0.95
            # Also bounded by half the cell size
            r_max = min(r_max, cell_w / 2.0 * 0.9, cell_h / 2.0 * 0.9)
            r = max(0.01, r_max)

            x[idx, 0] = cx
            x[idx, 1] = cy
            x[idx, 2] = r

        return x

    # Try multiple aspect ratios and grid configurations
    aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.2, 0.8, 1.1, 0.9]
    grid_configs = [(5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (4, 5), (5, 4)]

    for ar in aspect_ratios:
        if time.time() - start_time > timeout:
            break

        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 - w

        if w < 0.4 or h < 0.4:
            continue

        for nx, ny in grid_configs:
            if time.time() - start_time > timeout:
                break

            # Generate initial placement
            x = grid_cell_placement(w, h, nx, ny)

            # Force relaxation
            x = force_relax(x, w, h, max_iter=200)

            # NLP polish
            x = nlp_polish(x, w, h, max_iter=150)

            # Evaluate
            score, valid = evaluate_solution(x, w, h)
            if valid and score > best_score:
                best_score = score
                best_solution = x.copy()
                best_w = w
                best_h = h

    # Additional random starts
    rng = np.random.default_rng(42)
    for trial in range(15):
        if time.time() - start_time > timeout:
            break

        w = rng.uniform(0.6, 1.4)
        h = 2.0 - w

        x = np.zeros((N, 3))
        x[:, 0] = rng.uniform(0.1, w - 0.1, N)
        x[:, 1] = rng.uniform(0.1, h - 0.1, N)
        x[:, 2] = rng.uniform(0.05, 0.15, N)

        x = force_relax(x, w, h, max_iter=200)
        x = nlp_polish(x, w, h, max_iter=150)

        score, valid = evaluate_solution(x, w, h)
        if valid and score > best_score:
            best_score = score
            best_solution = x.copy()
            best_w = w
            best_h = h

    # Fallback if no valid solution
    if best_solution is None:
        w = 1.0
        h = 1.0
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_solution = np.array(cells[:N])
        best_w = w
        best_h = h

    return best_solution
# EVOLVE-BLOCK-END
