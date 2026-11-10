import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 20.0
    N = 16

    def get_score(pts):
        """Compute R = (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def get_cost(pts):
        """Return 1/R for minimization. Lower is better."""
        s = get_score(pts)
        if s < 1e-12:
            return np.inf
        return 1.0 / s

    def get_cost_vec(params):
        pts = params.reshape(16, 2)
        return get_cost(pts)

    def smooth_obj_and_grad(params, k):
        """
        Smooth objective: minimize soft_max(d) / soft_min(d) where d are pairwise distances.
        """
        pts = params.reshape(16, 2)
        d = pdist(pts)
        d = np.maximum(d, 1e-12)

        kd = k * d
        kd_max = np.max(kd)
        kd_min = np.min(kd)

        soft_max = (1.0 / k) * (kd_max + np.log(np.sum(np.exp(kd - kd_max))))
        soft_min = -(1.0 / k) * (kd_min + np.log(np.sum(np.exp(-kd - kd_min))))

        if soft_min < 1e-12:
            return 1e10, np.zeros_like(params)

        cost = soft_max / soft_min

        exp_kd = np.exp(kd - kd_max)
        sum_exp_kd = np.sum(exp_kd)
        grad_soft_max = exp_kd / sum_exp_kd

        neg_kd = -kd - kd_min
        exp_neg_kd = np.exp(neg_kd)
        sum_exp_neg_kd = np.sum(exp_neg_kd)
        grad_soft_min = -exp_neg_kd / sum_exp_neg_kd

        d_cost_d_d = (grad_soft_max * soft_min - soft_max * grad_soft_min) / (soft_min ** 2)

        n = 16
        pair_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                pair_idx.append((i, j))

        grad_pts = np.zeros((16, 2))
        for idx2, (a, b) in enumerate(pair_idx):
            if d[idx2] < 1e-12:
                continue
            diff = pts[a] - pts[b]
            grad_pts[a] += d_cost_d_d[idx2] * diff / d[idx2]
            grad_pts[b] += d_cost_d_d[idx2] * (-diff) / d[idx2]

        return cost, grad_pts.flatten()

    def smooth_obj(params, k):
        c, _ = smooth_obj_and_grad(params, k)
        return c

    def smooth_grad(params, k):
        _, g = smooth_obj_and_grad(params, k)
        return g

    def hexagonal_init():
        """Create 16 points from a hexagonal lattice subset."""
        a = 1.0
        points = []
        for row in range(-3, 4):
            for col in range(-3, 4):
                x = col * a + (row % 2) * a / 2
                y = row * a * np.sqrt(3) / 2
                points.append([x, y])
        points = np.array(points)
        dists = np.sum(points ** 2, axis=1)
        sorted_idx = np.argsort(dists)
        sel = points[sorted_idx[:16]]
        sel = sel - np.mean(sel, axis=0)
        return sel

    def regular_polygon_init(n, radius=1.0):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = radius * np.column_stack([np.cos(angles), np.sin(angles)])
        pts = pts - np.mean(pts, axis=0)
        return pts

    def generate_initial_configs():
        """Generate diverse initial configurations."""
        configs = []

        # Two octagons with various parameters
        for ratio in [1.2, 1.4, 1.5, 1.6, 1.8, 2.0]:
            for phase in [0.0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]:
                n = 8
                inner_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
                outer_angles = inner_angles + phase
                inner_pts = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
                outer_pts = ratio * np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
                pts = np.vstack([inner_pts, outer_pts])
                pts = pts - np.mean(pts, axis=0)
                configs.append(pts)

        # Hexagonal lattice subset
        configs.append(hexagonal_init())

        # 4x4 grid
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        configs.append(g)

        # Regular 16-gon
        configs.append(regular_polygon_init(16))

        # Three rings: 4 + 8 + 4
        for r1, r2 in [(0.5, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0)]:
            a4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
            a8 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
            pts1 = r1 * np.column_stack([np.cos(a4), np.sin(a4)])
            pts2 = r2 * np.column_stack([np.cos(a8), np.sin(a8)])
            pts3 = r1 * np.column_stack([np.cos(a4 + np.pi/4), np.sin(a4 + np.pi/4)])
            pts = np.vstack([pts1, pts2, pts3])
            pts = pts - np.mean(pts, axis=0)
            configs.append(pts)

        # Four rings: 4 + 4 + 4 + 4 with offsets
        for r1, r2, r3 in [(0.3, 0.6, 1.0), (0.4, 0.7, 1.0), (0.5, 0.8, 1.0)]:
            a4a = np.linspace(0, 2*np.pi, 4, endpoint=False)
            a4b = a4a + np.pi/4
            p1 = r1 * np.column_stack([np.cos(a4a), np.sin(a4a)])
            p2 = r2 * np.column_stack([np.cos(a4b), np.sin(a4b)])
            p3 = r3 * np.column_stack([np.cos(a4a), np.sin(a4a)])
            p4 = r3 * np.column_stack([np.cos(a4b), np.sin(a4b)])
            pts = np.vstack([p1, p2, p3, p4])
            pts = pts - np.mean(pts, axis=0)
            configs.append(pts)

        # Random initializations
        rng = np.random.default_rng(42)
        for _ in range(20):
            pts = rng.normal(size=(16, 2))
            pts = pts - np.mean(pts, axis=0)
            configs.append(pts)

        return configs

    def simulated_annealing(pts, max_iter, T0, T1, rng):
        """
        Simulated annealing to escape local minima.
        Uses random perturbations with temperature schedule.
        """
        current = pts.copy()
        current_score = get_score(current)
        best = current.copy()
        best_score = current_score

        for it in range(max_iter):
            # Temperature schedule: linear from T0 to T1
            T = T0 * (T1 / T0) ** (it / max_iter)

            # Generate perturbation
            perturbation_scale = 0.1 * T
            delta = rng.normal(scale=perturbation_scale, size=(16, 2))
            candidate = current + delta

            candidate_score = get_score(candidate)

            # Accept if better or with probability exp(-delta_score/T)
            if candidate_score >= current_score:
                current = candidate
                current_score = candidate_score
            else:
                delta_score = current_score - candidate_score
                if delta_score > 0:
                    accept_prob = np.exp(-delta_score / max(T, 1e-10))
                    if rng.random() < accept_prob:
                        current = candidate
                        current_score = candidate_score

            # Update best
            if current_score > best_score:
                best = current.copy()
                best_score = current_score

            # Check time
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break

        return best, best_score

    def gradient_polish(pts, k_values, max_iter_each):
        """Apply gradient-based optimization with increasing k."""
        current = pts.copy()
        current_score = get_score(current)

        for k in k_values:
            try:
                res = minimize(
                    lambda p: smooth_obj(p, k),
                    current.flatten(),
                    jac=lambda p: smooth_grad(p, k),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_each, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 50}
                )
                if np.isfinite(res.fun):
                    res_pts = res.x.reshape(16, 2)
                    res_score = get_score(res_pts)
                    if res_score > current_score:
                        current = res_pts
                        current_score = res_score
            except Exception:
                pass

            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break

        return current, current_score

    # Generate initial configurations
    initial_configs = generate_initial_configs()

    # Find best initial configuration
    best_pts = None
    best_score = 0.0
    for cfg in initial_configs:
        s = get_score(cfg)
        if s > best_score:
            best_score = s
            best_pts = cfg.copy()

    # Phase 1: Simulated annealing from multiple starts
    rng = np.random.default_rng(123)
    time_elapsed = time.time() - start_time

    if (TIME_LIMIT - time_elapsed) > 120 and best_pts is not None:
        # Run SA from top 5 initial configs
        scored_configs = [(get_score(cfg), cfg) for cfg in initial_configs]
        scored_configs.sort(key=lambda x: -x[0])
        top_configs = [cfg for _, cfg in scored_configs[:5]]

        for cfg in top_configs:
            if time.time() - start_time > TIME_LIMIT - 30:
                break

            sa_pts, sa_score = simulated_annealing(
                cfg,
                max_iter=2000,
                T0=0.5,
                T1=0.01,
                rng=rng
            )

            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts

    # Phase 2: Gradient polish on the best SA result
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 60 and best_pts is not None:
        polished_pts, polished_score = gradient_polish(
            best_pts,
            k_values=[20, 50, 100, 200, 500],
            max_iter_each=500
        )
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Phase 3: Another round of SA + polish from the polished result
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 40 and best_pts is not None:
        sa_pts2, sa_score2 = simulated_annealing(
            best_pts,
            max_iter=1000,
            T0=0.3,
            T1=0.005,
            rng=rng
        )
        if sa_score2 > best_score:
            best_score = sa_score2
            best_pts = sa_pts2

        polished_pts2, polished_score2 = gradient_polish(
            best_pts,
            k_values=[100, 200, 500, 1000],
            max_iter_each=300
        )
        if polished_score2 > best_score:
            best_score = polished_score2
            best_pts = polished_pts2

    # Phase 4: Final Nelder-Mead polish
    time_elapsed = time.time() - start_time
    if (TIME_LIMIT - time_elapsed) > 20 and best_pts is not None:
        try:
            res3 = minimize(get_cost_vec, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if np.isfinite(res3.fun):
                res_pts3 = res3.x.reshape(16, 2)
                res_score3 = get_score(res_pts3)
                if res_score3 > best_score:
                    best_pts = res_pts3
                    best_score = res_score3
        except Exception:
            pass

    final_pts = best_pts
    if final_pts is None or not np.all(np.isfinite(final_pts)):
        g = np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
        g = g - np.mean(g, axis=0)
        final_pts = g

    return final_pts
