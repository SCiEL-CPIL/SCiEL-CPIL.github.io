import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 145.0  # Leave 5s margin for safety

    # Precompute indices for all triples
    triples = list(combinations(range(N), 3))
    n_triples = len(triples)

    i_idx = np.array([t[0] for t in triples])
    j_idx = np.array([t[1] for t in triples])
    k_idx = np.array([t[2] for t in triples])

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = np.min(areas)
        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """Smooth surrogate using log-sum-exp for min triangle area."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        score = soft_min_area / area_hull
        return -score

    def _hard_score_neg(points_flat):
        """Negative hard score for final evaluation."""
        points = points_flat.reshape(-1, 2)
        score, _ = compute_score(points)
        return -score

    def _soft_objective(points_flat, beta=50.0):
        """Soft-min objective for gradient-based optimization."""
        points = points_flat.reshape(-1, 2)
        return _soft_min_score(points, beta)

    def _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner):
        """
        Generate points with two concentric regular polygons.
        n_hull + n_inner = 13
        """
        angles_hull = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
        angles_inner = np.linspace(rot_inner, rot_inner + 2*np.pi, n_inner, endpoint=False)
        
        xs = np.concatenate([r_hull * np.cos(angles_hull), r_inner * np.cos(angles_inner)])
        ys = np.concatenate([r_hull * np.sin(angles_hull), r_inner * np.sin(angles_inner)])
        return np.column_stack([xs, ys])

    def _symmetric_objective(params, n_hull, n_inner, beta=50.0):
        """
        Objective for symmetric parameterization.
        params = [r_hull, r_inner, rot_hull, rot_inner]
        """
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        return _soft_min_score(points, beta)

    def _symmetric_hard_objective(params, n_hull, n_inner):
        """Hard score for symmetric parameterization."""
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        score, _ = compute_score(points)
        return -score

    best_points = None
    best_score = -float('inf')

    # Strategy 1: Optimize symmetric configurations (k-gon + m-gon)
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]

    for n_hull, n_inner in splits:
        if time.time() > start_time + time_limit - 30:
            break

        # Initial guess
        r_hull_init = 1.0
        r_inner_init = 0.4 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0

        # Phase 1: Coarse optimization with soft-min
        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 30:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: _symmetric_objective(x, nh, ni, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = result.x
                points = _make_symmetric_points(n_hull, n_inner, *x0)
                s, _ = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            except Exception:
                pass

        # Phase 2: Fine-tune with hard score using Nelder-Mead
        if time.time() > start_time + time_limit - 30:
            break

        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: _symmetric_hard_objective(x, nh, ni),
                x0,
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            x0 = res.x
            points = _make_symmetric_points(n_hull, n_inner, *x0)
            s, _ = compute_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 2: Multi-start random perturbations from best symmetric result
    if best_points is not None and time.time() < start_time + time_limit - 20:
        base = best_points.copy()
        n_perturb = 8
        for _ in range(n_perturb):
            if time.time() > start_time + time_limit - 20:
                break
            
            # Random perturbation
            perturbed = base + np.random.normal(0, 0.03, base.shape)
            s, _ = compute_score(perturbed)
            if s > best_score:
                best_score = s
                best_points = perturbed.copy()
            
            # Optimize from perturbed point
            current = perturbed.copy()
            for beta in [50.0, 100.0]:
                if time.time() > start_time + time_limit - 20:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 60, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    optimized = result.x.reshape(-1, 2)
                    opt_score, _ = compute_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
                    current = optimized
                except Exception:
                    pass

    # Strategy 3: Full free-space optimization from best result
    if best_points is not None and time.time() < start_time + time_limit - 15:
        current = best_points.copy()
        
        # Multi-beta annealing in free space
        for beta in [30.0, 80.0, 150.0]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                optimized = result.x.reshape(-1, 2)
                opt_score, _ = compute_score(optimized)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = optimized.copy()
                current = optimized
            except Exception:
                pass

    # Strategy 4: Random restarts in free space
    if time.time() < start_time + time_limit - 10:
        np.random.seed(42)
        n_restarts = 4
        for _ in range(n_restarts):
            if time.time() > start_time + time_limit - 10:
                break
            
            # Generate random configuration
            n_hull = np.random.randint(8, 13)
            n_inner = N - n_hull
            r_hull = np.random.uniform(0.9, 1.2)
            r_inner = np.random.uniform(0.2, 0.5) if n_inner > 0 else 0.0
            rot_hull = np.random.uniform(0, 2*np.pi)
            rot_inner = np.random.uniform(0, 2*np.pi) if n_inner > 1 else 0.0
            
            config = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
            config = config + np.random.normal(0, 0.05, config.shape)

            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()

            # Quick optimization
            if time.time() > start_time + time_limit - 10:
                break

            current = config.copy()
            for beta in [50.0, 100.0]:
                if time.time() > start_time + time_limit - 10:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta: _soft_objective(x, b),
                        current.flatten(),
                        method='L-BFGS-B',
                        options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    optimized = result.x.reshape(-1, 2)
                    opt_score, _ = compute_score(optimized)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_points = optimized.copy()
                    current = optimized
                except Exception:
                    pass

    # Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 5:
        try:
            result = opt.minimize(
                lambda x: _soft_objective(x, 200.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished = result.x.reshape(-1, 2)
            pol_score, _ = compute_score(polished)
            if pol_score > best_score:
                best_score = pol_score
                best_points = polished.copy()
        except Exception:
            pass

        if time.time() < start_time + time_limit - 2:
            try:
                res = opt.minimize(
                    _hard_score_neg,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                final_points = res.x.reshape(-1, 2)
                final_score, _ = compute_score(final_points)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_points.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: regular 13-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
