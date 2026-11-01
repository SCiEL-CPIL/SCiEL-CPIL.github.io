"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32
PENALTY = 200.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    best_score = -np.inf
    best_result = None

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

    def _make_random_init(seed):
        """Create random initialization."""
        rng = np.random.RandomState(seed)
        return rng.uniform(0.1, 0.9, size=(N, 2))

    def _compute_initial_radius(pts):
        """Compute initial radius based on minimum pairwise distance."""
        min_dist = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist:
                    min_dist = d
        return min_dist / 2 * 0.85

    def _objective(vars_):
        """Objective: minimize -sum(radii) + penalty for violations."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        # Pairwise distances (vectorized)
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        r_sum = R[:, None] + R[None, :]
        overlap = np.maximum(r_sum - d, 0.0)
        pen = np.sum(overlap * overlap)
        # Boundary violations
        bnd = np.maximum(
            np.maximum(np.maximum(
                R - X, X + R - 1.0),
                R - Y),
            Y + R - 1.0
        )
        pen += np.sum(bnd * bnd)
        return obj + PENALTY * pen

    def _objective_grad(vars_):
        """Analytical gradient of the objective (vectorized)."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        weight = PENALTY
        # Pairwise
        dx = X[:, None] - X[None, :]
        dy = Y[:, None] - Y[None, :]
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        r_sum = R[:, None] + R[None, :]
        overlap = np.maximum(r_sum - d, 0.0)
        mask = overlap > 0
        # d(overlap)/dX[i] = -dx[i,j]/d[i,j] for j where overlap>0
        # grad_X[i] += weight * 2*overlap[i,j] * (-dx[i,j]/d[i,j])
        # But we need to sum over j. Use matrix ops.
        inv_d = np.where(mask, 1.0 / np.where(d > 1e-10, d, 1.0), 0.0)
        factor = 2.0 * overlap * inv_d  # shape (N,N)
        # grad_X[i] = sum_j weight * factor[i,j] * (-dx[i,j])
        # = -weight * sum_j factor[i,j] * dx[i,j]
        grad_x_pair = -weight * np.sum(factor * dx, axis=1)
        grad_y_pair = -weight * np.sum(factor * dy, axis=1)
        # grad_R[i] = sum_j weight * 2*overlap[i,j] (from pairwise)
        grad_r_pair = weight * np.sum(2.0 * overlap, axis=1)
        grad[0::3] += grad_x_pair
        grad[1::3] += grad_y_pair
        grad[2::3] += grad_r_pair
        # Boundary
        # val = X - R (left), 1-X-R (right), Y-R (bottom), 1-Y-R (top)
        # For each, if val < 0: pen += val^2, dpen/dX = 2*val * d(val)/dX
        v_left = X - R
        m = v_left < 0
        grad[0::3] += weight * np.where(m, 2.0 * v_left, 0.0)
        grad[2::3] += weight * np.where(m, -2.0 * v_left, 0.0)
        v_right = 1.0 - X - R
        m = v_right < 0
        grad[0::3] += weight * np.where(m, -2.0 * v_right, 0.0)
        grad[2::3] += weight * np.where(m, -2.0 * v_right, 0.0)
        v_bot = Y - R
        m = v_bot < 0
        grad[1::3] += weight * np.where(m, 2.0 * v_bot, 0.0)
        grad[2::3] += weight * np.where(m, -2.0 * v_bot, 0.0)
        v_top = 1.0 - Y - R
        m = v_top < 0
        grad[1::3] += weight * np.where(m, -2.0 * v_top, 0.0)
        grad[2::3] += weight * np.where(m, -2.0 * v_top, 0.0)
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

    def _polish(pos, rad, n_iters=10):
        """Iteratively recompute max radii for given positions (vectorized)."""
        pos = pos.copy()
        rad = rad.copy()
        for _ in range(n_iters):
            # Compute pairwise distances
            dx = pos[:, 0:1] - pos[None, :, 0]
            dy = pos[:, 1:2] - pos[None, :, 1]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            # For circle i, constraint from circle j: r_i <= d[i,j] - r_j
            # So max r_i = min_j (d[i,j] - r_j), also bounded by walls
            constraints = d - rad[None, :]  # shape (N, N), constraints[i,j] = d_ij - r_j
            max_r_pairs = np.min(constraints, axis=1)
            # Wall constraints
            wall = np.minimum(
                np.minimum(pos[:, 0], 1.0 - pos[:, 0]),
                np.minimum(pos[:, 1], 1.0 - pos[:, 1])
            )
            rad = np.maximum(1e-6, np.minimum(max_r_pairs, wall) - 1e-8)
        return rad

    def _high_precision_polish(pos, rad, time_budget=5.0):
        """High-precision polish using SLSQP on a smooth epigraph formulation.
        Variables: x_i, y_i, r_i for each circle.
        Maximize sum(r_i) subject to:
          x_i + r_i <= 1, x_i - r_i >= 0, y_i + r_i <= 1, y_i - r_i >= 0
          (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i<j
        We use a smooth approximation: minimize -sum(r) + soft penalty.
        """
        start = time.time()
        pos = pos.copy().astype(float)
        rad = rad.copy().astype(float)

        def hp_objective(vars_):
            X = vars_[0::3]
            Y = vars_[1::3]
            R = vars_[2::3]
            obj = -np.sum(R)
            # Smooth pairwise: use softplus-like for overlap
            dx = X[:, None] - X[None, :]
            dy = Y[:, None] - Y[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            r_sum = R[:, None] + R[None, :]
            overlap = np.maximum(r_sum - d, 0.0)
            pen = np.sum(overlap * overlap)
            # Boundary
            bnd = np.maximum(np.maximum(np.maximum(R - X, X + R - 1.0),
                                        R - Y), Y + R - 1.0)
            pen += np.sum(bnd * bnd)
            return obj + 500.0 * pen

        def hp_grad(vars_):
            X = vars_[0::3]
            Y = vars_[1::3]
            R = vars_[2::3]
            grad = np.zeros_like(vars_)
            grad[2::3] = -1.0
            w = 500.0
            dx = X[:, None] - X[None, :]
            dy = Y[:, None] - Y[None, :]
            d = np.sqrt(dx * dx + dy * dy)
            np.fill_diagonal(d, np.inf)
            r_sum = R[:, None] + R[None, :]
            overlap = np.maximum(r_sum - d, 0.0)
            mask = overlap > 0
            inv_d = np.where(mask, 1.0 / np.where(d > 1e-10, d, 1.0), 0.0)
            factor = 2.0 * overlap * inv_d
            grad[0::3] += -w * np.sum(factor * dx, axis=1)
            grad[1::3] += -w * np.sum(factor * dy, axis=1)
            grad[2::3] += w * np.sum(2.0 * overlap, axis=1)
            v = X - R; m = v < 0
            grad[0::3] += w * np.where(m, 2.0 * v, 0.0)
            grad[2::3] += w * np.where(m, -2.0 * v, 0.0)
            v = 1.0 - X - R; m = v < 0
            grad[0::3] += w * np.where(m, -2.0 * v, 0.0)
            grad[2::3] += w * np.where(m, -2.0 * v, 0.0)
            v = Y - R; m = v < 0
            grad[1::3] += w * np.where(m, 2.0 * v, 0.0)
            grad[2::3] += w * np.where(m, -2.0 * v, 0.0)
            v = 1.0 - Y - R; m = v < 0
            grad[1::3] += w * np.where(m, -2.0 * v, 0.0)
            grad[2::3] += w * np.where(m, -2.0 * v, 0.0)
            return grad

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = pos[i, 0]
            x0[3 * i + 1] = pos[i, 1]
            x0[3 * i + 2] = rad[i]

        hp_bounds = []
        for i in range(N):
            hp_bounds.append((0.0, 1.0))
            hp_bounds.append((0.0, 1.0))
            hp_bounds.append((1e-6, 0.5))

        try:
            res = minimize(hp_objective, x0, method='L-BFGS-B', jac=hp_grad,
                           bounds=hp_bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x
            pos_new = np.column_stack((x_opt[0::3], x_opt[1::3]))
            rad_new = x_opt[2::3].copy()
            # Final radius polish
            rad_new = _polish(pos_new, rad_new, n_iters=15)
            score_new = _evaluate(pos_new, rad_new)
            score_old = _evaluate(pos, rad)
            if score_new > score_old:
                return pos_new, rad_new
        except Exception:
            pass
        return pos, rad

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((1e-5, 0.5))

    # Define initializations: hex, grid, and several random
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(20):
        inits.append(('rand', _make_random_init(seed + 42)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if time.time() > deadline - 3:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # For later restarts, use fewer iterations to save time
        maxiter = 400 if idx < 3 else 200

        try:
            res = minimize(_objective, x0, method='L-BFGS-B', jac=_objective_grad,
                           bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-11, 'gtol': 1e-9})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Polish radii
        R = _polish(pos, R, n_iters=8)

        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # High-precision polish on the best result if time permits
    if time.time() < deadline - 8:
        bp_pos = best_result[:, 0:2].copy()
        bp_rad = best_result[:, 2].copy()
        bp_pos, bp_rad = _high_precision_polish(bp_pos, bp_rad, time_budget=8.0)
        bp_score = _evaluate(bp_pos, bp_rad)
        if bp_score > best_score:
            best_score = bp_score
            best_result = np.column_stack([bp_pos, bp_rad])

    return best_result
