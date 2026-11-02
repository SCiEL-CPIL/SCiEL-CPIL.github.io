"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def _make_hex_init():
        """Create hexagonal lattice initialization."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        return init_pts

    def _make_grid_init():
        """Create grid initialization."""
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([0.5/side + i/side, 0.5/side + j/side])
        return np.array(centers[:N])

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2 * 0.85

    def _objective(vars_, penalty_weight=500.0):
        """Objective: minimize -sum(radii) + penalty for violations."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    pen += overlap ** 2
            if X[i] - R[i] < 0:
                pen += (X[i] - R[i]) ** 2
            if 1.0 - X[i] - R[i] < 0:
                pen += (1.0 - X[i] - R[i]) ** 2
            if Y[i] - R[i] < 0:
                pen += (Y[i] - R[i]) ** 2
            if 1.0 - Y[i] - R[i] < 0:
                pen += (1.0 - Y[i] - R[i]) ** 2
        return obj + penalty_weight * pen

    def _objective_grad(vars_, penalty_weight=500.0):
        """Analytical gradient of the objective."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = penalty_weight
        for i in range(N):
            for j in range(i + 1, N):
                dx = X[i] - X[j]
                dy = Y[i] - Y[j]
                d = np.sqrt(dx * dx + dy * dy)
                if d < 1e-10:
                    continue
                overlap = R[i] + R[j] - d
                if overlap > 0:
                    factor = 2.0 * overlap
                    grad[3 * i] += weight * factor * (-dx / d)
                    grad[3 * i + 1] += weight * factor * (-dy / d)
                    grad[3 * j] += weight * factor * (dx / d)
                    grad[3 * j + 1] += weight * factor * (dy / d)
                    grad[3 * i + 2] += weight * factor
                    grad[3 * j + 2] += weight * factor
            val = X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - X[i] - R[i]
            if val < 0:
                grad[3 * i] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * 2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
            val = 1.0 - Y[i] - R[i]
            if val < 0:
                grad[3 * i + 1] += weight * -2.0 * val
                grad[3 * i + 2] += weight * -2.0 * val
        return grad

    def _evaluate(centers, radii):
        """Compute sum of radii if valid, else -inf."""
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9:
                return -np.inf
            if y - r < -1e-9 or y + r > 1 + 1e-9:
                return -np.inf
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-9:
                    return -np.inf
        return np.sum(radii)

    def _polish(pos, rad, n_iters=50):
        """Iteratively recompute max radii for given positions."""
        for _ in range(n_iters):
            changed = False
            for i in range(N):
                min_dist_others = np.inf
                for j in range(N):
                    if i == j:
                        continue
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist_others = min(min_dist_others, d - rad[j])
                min_dist_bound = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
                new_r = max(1e-8, min(min_dist_others, min_dist_bound) - 1e-8)
                if abs(new_r - rad[i]) > 1e-12:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    best_score = -np.inf
    best_result = None

    # Initializations
    inits = []
    inits.append(_make_hex_init())
    inits.append(_make_grid_init())
    rng = np.random.RandomState(42)
    for _ in range(5):
        inits.append(rng.uniform(0.1, 0.9, size=(N, 2)))

    for init_pts in inits:
        if time.time() > deadline - 10:
            break

        init_r = _compute_initial_radius(init_pts)
        
        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # Use Basin Hopping for structural global search
        # It runs L-BFGS-B locally and perturbs globally
        try:
            res = basinhopping(
                _objective, 
                x0, 
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": bounds, "jac": _objective_grad},
                niter=10, 
                stepsize=0.2, 
                T=100.0, 
                accept_test=None,
                minimizer='L-BFGS-B'
            )
            
            # Extract result
            centers = res.x[0::3]
            radii = res.x[2::3]
            
            # Polish
            radii = _polish(centers, radii.copy())
            
            score = _evaluate(centers, radii)
            
            if score > best_score:
                best_score = score
                best_result = np.column_stack((centers, radii))
                
        except Exception:
            pass

    # If no valid result found via optimization, fallback to grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
        # Polish the grid too
        centers = best_result[:, :2]
        radii = best_result[:, 2].copy()
        radii = _polish(centers, radii)
        best_result[:, 2] = radii

    return best_result
