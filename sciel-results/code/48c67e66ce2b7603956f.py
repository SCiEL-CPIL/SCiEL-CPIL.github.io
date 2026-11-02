<<<<<<< SEARCH
"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    def _elapsed():
        return time.time() - start_time

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
        """Iteratively recompute max radii for given positions with more iterations."""
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
                new_r = max(1e-6, min(min_dist_others, min_dist_bound) - 1e-6)
                if abs(new_r - rad[i]) > 1e-10:
                    changed = True
                rad[i] = new_r
            if not changed:
                break
        return rad

    def _high_precision_polish(pos, rad, n_iters=100):
        """High-precision polish with tighter tolerances."""
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

    def _local_center_optimize(pos, rad, n_iters=200):
        """
        Optimize positions only (radii fixed) to minimize overlaps.
        Uses gradient descent on a smooth objective.
        """
        # We'll do a simple coordinate descent / gradient step on positions
        # Objective: sum of squared overlaps (pairwise + boundary)
        # For fixed radii, we want to maximize min distances
        
        # Use L-BFGS-B on positions only with the same penalty structure
        def pos_obj(p):
            X = p[0::2]
            Y = p[1::2]
            pen = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    dx = X[i] - X[j]
                    dy = Y[i] - Y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    overlap = rad[i] + rad[j] - d
                    if overlap > 0:
                        pen += overlap ** 2
                if X[i] - rad[i] < 0:
                    pen += (X[i] - rad[i]) ** 2
                if 1.0 - X[i] - rad[i] < 0:
                    pen += (1.0 - X[i] - rad[i]) ** 2
                if Y[i] - rad[i] < 0:
                    pen += (Y[i] - rad[i]) ** 2
                if 1.0 - Y[i] - rad[i] < 0:
                    pen += (1.0 - Y[i] - rad[i]) ** 2
            return pen
        
        def pos_grad(p):
            X = p[0::2]
            Y = p[1::2]
            grad = np.zeros_like(p)
            for i in range(N):
                for j in range(i + 1, N):
                    dx = X[i] - X[j]
                    dy = Y[i] - Y[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < 1e-10:
                        continue
                    overlap = rad[i] + rad[j] - d
                    if overlap > 0:
                        factor = 2.0 * overlap
                        grad[2 * i] += factor * (-dx / d)
                        grad[2 * i + 1] += factor * (-dy / d)
                        grad[2 * j] += factor * (dx / d)
                        grad[2 * j + 1] += factor * (dy / d)
                val = X[i] - rad[i]
                if val < 0:
                    grad[2 * i] += 2.0 * val
                val = 1.0 - X[i] - rad[i]
                if val < 0:
                    grad[2 * i] += -2.0 * val
                val = Y[i] - rad[i]
                if val < 0:
                    grad[2 * i + 1] += 2.0 * val
                val = 1.0 - Y[i] - rad[i]
                if val < 0:
                    grad[2 * i + 1] += -2.0 * val
            return grad
        
        p0 = np.zeros(2 * N)
        p0[0::2] = pos[:, 0]
        p0[1::2] = pos[:, 1]
        
        bounds_p = []
        for i in range(N):
            bounds_p.append((0.0, 1.0))
            bounds_p.append((0.0, 1.0))
        
        try:
            res = minimize(pos_obj, p0, method='L-BFGS-B',
                           jac=pos_grad,
                           bounds=bounds_p,
                           options={'maxiter': n_iters, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            p_opt = res.x
            pos_new = np.column_stack((p_opt[0::2], p_opt[1::2]))
            return pos_new
        except Exception:
            return pos.copy()

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
        if _elapsed() > deadline - 5:
            break

        init_r = _compute_initial_radius(init_pts)

        x0 = np.zeros(3 * N)
        for i in range(N):
            x0[3 * i] = init_pts[i, 0]
            x0[3 * i + 1] = init_pts[i, 1]
            x0[3 * i + 2] = init_r

        # Use adaptive penalty: start high, the optimizer will handle it
        # For early restarts, use more iterations
        maxiter = 500 if idx < 3 else 200

        try:
            res = minimize(_objective, x0, method='L-BFGS-B', 
                           jac=lambda v: _objective_grad(v, 500.0),
                           bounds=bounds, 
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxls': 50})
            x_opt = res.x
        except Exception:
            x_opt = x0

        X = x_opt[0::3].copy()
        Y = x_opt[1::3].copy()
        R = x_opt[2::3].copy()
        pos = np.column_stack((X, Y))

        # Polish radii
        R = _polish(pos, R, n_iters=50)

        # Local center optimization with fixed radii to improve packing
        if _elapsed() < deadline - 10:
            pos = _local_center_optimize(pos, R, n_iters=100)
            R = _polish(pos, R, n_iters=50)

        score = _evaluate(pos, R)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([pos, R])

    # Final high-precision polish on the best result
    if best_result is not None:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        
        # Do a final local center optimization
        if _elapsed() < deadline - 5:
            pos = _local_center_optimize(pos, rad, n_iters=200)
            rad = _polish(pos, rad, n_iters=50)
        
        rad = _high_precision_polish(pos, rad, n_iters=100)
        final_score = _evaluate(pos, rad)
        if final_score > best_score:
            best_score = final_score
            best_result = np.column_stack([pos, rad])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
=======
"""Circle packing in unit square n=32: maximize sum of radii."""
import numpy as np
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    best_score = -np.inf
    best_result = None

    def _elapsed():
        return time.time() - start_time

    def _compute_max_radii(pos):
        """For given positions, compute the maximum feasible radius for each circle."""
        rad = np.zeros(N)
        for i in range(N):
            min_dist = min(pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            for j in range(N):
                if i != j:
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist = min(min_dist, d / 2)
            rad[i] = max(1e-8, min_dist - 1e-8)
        return rad

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
        return rng.uniform(0.05, 0.95, size=(N, 2))

    def _simulated_annealing(pos0, n_iter=50000, t_start=0.1, t_end=0.001, rng_seed=42):
        """
        Simulated annealing on center positions.
        Objective: maximize sum of max feasible radii.
        """
        rng = np.random.RandomState(rng_seed)
        pos = pos0.copy()
        rad = _compute_max_radii(pos)
        score = np.sum(rad)
        best_pos = pos.copy()
        best_rad = rad.copy()
        best_score = score

        n_steps = n_iter
        for it in range(n_steps):
            if _elapsed() > deadline - 5:
                break
            alpha = it / max(1, n_steps - 1)
            temperature = t_start * (t_end / t_start) ** alpha
            step_size = 0.02 * (1 - alpha * 0.7)

            # Random perturbation
            i = rng.randint(N)
            delta = rng.normal(0, step_size, size=2)
            new_pos = pos.copy()
            new_pos[i] = np.clip(new_pos[i] + delta, 0.01, 0.99)
            
            new_rad = _compute_max_radii(new_pos)
            new_score = np.sum(new_rad)
            delta_score = new_score - score

            if delta_score > 0 or rng.random() < np.exp(delta_score / max(temperature, 1e-10)):
                pos = new_pos
                rad = new_rad
                score = new_score
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_rad = rad.copy()

        return best_pos, best_rad

    def _local_refinement(pos, rad, n_iters=2000):
        """
        Gradient-free local refinement using coordinate descent.
        For each circle, try small moves in x and y to improve the total radius sum.
        """
        pos = pos.copy()
        rad = _compute_max_radii(pos)
        score = np.sum(rad)
        best_pos = pos.copy()
        best_rad = rad.copy()
        best_score = score
        
        rng = np.random.RandomState(123)
        step_sizes = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for it in range(n_iters):
            if _elapsed() > deadline - 5:
                break
            improved = False
            # Shuffle circle order
            order = rng.permutation(N)
            for i in order:
                if _elapsed() > deadline - 5:
                    break
                # Try small moves in x and y
                for dx, dy in [(step_sizes[0], 0), (-step_sizes[0], 0), 
                               (0, step_sizes[0]), (0, -step_sizes[0]),
                               (step_sizes[1], 0), (-step_sizes[1], 0),
                               (0, step_sizes[1]), (0, -step_sizes[1])]:
                    new_pos = pos.copy()
                    new_pos[i] = np.clip(new_pos[i] + [dx, dy], 0.01, 0.99)
                    new_rad = _compute_max_radii(new_pos)
                    new_score = np.sum(new_rad)
                    if new_score > score + 1e-10:
                        pos = new_pos
                        rad = new_rad
                        score = new_score
                        improved = True
                        break
            
            if not improved:
                # Try smaller steps
                step_sizes = [s * 0.5 for s in step_sizes]
                if step_sizes[0] < 1e-5:
                    break
            else:
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_rad = rad.copy()

        return best_pos, best_rad

    def _high_precision_polish(pos, rad, n_iters=200):
        """High-precision polish with tighter tolerances."""
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

    # Define initializations
    inits = []
    inits.append(('hex', _make_hex_init()))
    inits.append(('grid', _make_grid_init()))
    for seed in range(15):
        inits.append(('rand', _make_random_init(seed + 100)))

    for idx, (init_type, init_pts) in enumerate(inits):
        if _elapsed() > deadline - 10:
            break

        # Simulated annealing
        sa_pos, sa_rad = _simulated_annealing(init_pts, n_iter=30000, rng_seed=idx + 1000)
        score = _evaluate(sa_pos, sa_rad)
        if score > best_score:
            best_score = score
            best_result = np.column_stack([sa_pos, sa_rad])

        # Local refinement
        if _elapsed() < deadline - 10:
            lr_pos, lr_rad = _local_refinement(sa_pos, sa_rad, n_iters=1500)
            score = _evaluate(lr_pos, lr_rad)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([lr_pos, lr_rad])

    # Final high-precision polish on the best result
    if best_result is not None:
        pos = best_result[:, :2].copy()
        rad = best_result[:, 2].copy()
        rad = _high_precision_polish(pos, rad, n_iters=100)
        final_score = _evaluate(pos, rad)
        if final_score > best_score:
            best_score = final_score
            best_result = np.column_stack([pos, rad])

    # If no valid result found, return grid fallback
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return best_result
>>>>>>> REPLACE
