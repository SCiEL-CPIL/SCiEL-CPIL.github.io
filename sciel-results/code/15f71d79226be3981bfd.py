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

    # Hull cache: store last points and hull info to avoid recomputation
    _hull_cache = {'points': None, 'hull': None, 'area': 0.0, 'count': 0}

    def _get_hull_area(points):
        """Get convex hull area with caching."""
        if _hull_cache['points'] is not None:
            # Check if points changed significantly
            diff = np.max(np.abs(points - _hull_cache['points']))
            if diff < 1e-8:
                return _hull_cache['area']
        
        try:
            hull = ConvexHull(points)
            area = hull.volume
            _hull_cache['points'] = points.copy()
            _hull_cache['hull'] = hull
            _hull_cache['area'] = area
            _hull_cache['count'] += 1
            return area
        except Exception:
            _hull_cache['points'] = None
            _hull_cache['area'] = 0.0
            return 0.0

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        area_hull = _get_hull_area(points)
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
        area_hull = _get_hull_area(points)
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
        """Generate points with two concentric regular polygons."""
        angles_hull = np.linspace(rot_hull, rot_hull + 2*np.pi, n_hull, endpoint=False)
        angles_inner = np.linspace(rot_inner, rot_inner + 2*np.pi, n_inner, endpoint=False)
        
        xs = np.concatenate([r_hull * np.cos(angles_hull), r_inner * np.cos(angles_inner)])
        ys = np.concatenate([r_hull * np.sin(angles_hull), r_inner * np.sin(angles_inner)])
        return np.column_stack([xs, ys])

    def _symmetric_objective(params, n_hull, n_inner, beta=50.0):
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        return _soft_min_score(points, beta)

    def _symmetric_hard_objective(params, n_hull, n_inner):
        r_hull, r_inner, rot_hull, rot_inner = params
        if r_hull < 0.1 or r_inner < 0.0 or r_hull < r_inner:
            return 1e6
        points = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
        score, _ = compute_score(points)
        return -score

    def _get_min_triple(points):
        """Return indices of the triple with minimum area."""
        x = points[:, 0]
        y = points[:, 1]
        v1x = x[j_idx] - x[i_idx]
        v1y = y[j_idx] - y[i_idx]
        v2x = x[k_idx] - x[i_idx]
        v2y = y[k_idx] - y[i_idx]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_idx = np.argmin(areas)
        return triples[min_idx]

    def _fourier_points(coeffs):
        """Generate 13 points from Fourier coefficients."""
        n_freq = len(coeffs)
        theta = 2 * np.pi * np.arange(N) / N
        points = np.zeros((N, 2))
        
        for j in range(n_freq):
            a, b, phase = coeffs[j]
            freq = j + 1
            points[:, 0] += a * np.cos(freq * theta + phase) + b * np.sin(freq * theta + phase)
        
        return points

    def _fourier_objective(params, n_freq, beta=50.0):
        coeffs = params.reshape(-1, 3)
        points = _fourier_points(coeffs)
        return _soft_min_score(points, beta)

    def _fourier_hard_objective(params, n_freq):
        coeffs = params.reshape(-1, 3)
        points = _fourier_points(coeffs)
        score, _ = compute_score(points)
        return -score

    def _directed_local_search(points, max_iters=50, step_scale=0.01):
        """
        Structural change: Directed local search that specifically perturbs
        points involved in the minimum-area triangle. This is a fundamentally
        different search mechanism from gradient-based optimization.
        """
        current = points.copy()
        current_score, _ = compute_score(current)
        best_local = current.copy()
        best_local_score = current_score
        
        for it in range(max_iters):
            if time.time() > start_time + time_limit - 5:
                break
            
            # Find the worst triple
            worst = _get_min_triple(current)
            wi, wj, wk = worst
            
            # Try perturbing each of the 3 points in the worst triple
            improved = False
            for pt_idx in [wi, wj, wk]:
                # Try small random perturbations
                for trial in range(8):
                    if time.time() > start_time + time_limit - 5:
                        break
                    # Random direction
                    direction = np.random.randn(2)
                    direction /= (np.linalg.norm(direction) + 1e-15)
                    
                    # Multiple step sizes
                    for step in [step_scale * 0.5, step_scale, step_scale * 2]:
                        trial_points = current.copy()
                        trial_points[pt_idx] += direction * step
                        trial_score, _ = compute_score(trial_points)
                        
                        if trial_score > current_score + 1e-12:
                            current = trial_points
                            current_score = trial_score
                            if current_score > best_local_score:
                                best_local_score = current_score
                                best_local = current.copy()
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try larger random kicks to escape
                kick = np.random.randn(N, 2) * step_scale * 0.3
                kick_points = current + kick
                kick_score, _ = compute_score(kick_points)
                if kick_score > current_score:
                    current = kick_points
                    current_score = kick_score
                    if current_score > best_local_score:
                        best_local_score = current_score
                        best_local = current.copy()
                else:
                    break  # No improvement, stop
        
        return best_local, best_local_score

    best_points = None
    best_score = -float('inf')

    # Strategy 1: Optimize symmetric configurations (k-gon + m-gon)
    splits = [(13, 0), (12, 1), (11, 2), (10, 3), (9, 4), (8, 5), (7, 6)]

    for n_hull, n_inner in splits:
        if time.time() > start_time + time_limit - 40:
            break

        r_hull_init = 1.0
        r_inner_init = 0.4 if n_inner > 0 else 0.0
        rot_hull_init = 0.0
        rot_inner_init = np.pi / n_inner if n_inner > 1 else 0.0

        x0 = np.array([r_hull_init, r_inner_init, rot_hull_init, rot_inner_init])
        
        for beta in [20.0, 50.0, 100.0]:
            if time.time() > start_time + time_limit - 40:
                break
            try:
                result = opt.minimize(
                    lambda x, b=beta, nh=n_hull, ni=n_inner: _symmetric_objective(x, nh, ni, b),
                    x0,
                    method='L-BFGS-B',
                    bounds=[(0.3, 2.0), (0.0, 1.0), (-np.pi, np.pi), (-np.pi, np.pi)],
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                x0 = result.x
                points = _make_symmetric_points(n_hull, n_inner, *x0)
                s, _ = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            except Exception:
                pass

        # Fine-tune with hard score using Powell (structural change: derivative-free, handles non-smooth)
        if time.time() > start_time + time_limit - 40:
            break

        try:
            res = opt.minimize(
                lambda x, nh=n_hull, ni=n_inner: _symmetric_hard_objective(x, nh, ni),
                x0,
                method='Powell',
                options={'maxiter': 150, 'ftol': 1e-12, 'xtol': 1e-10}
            )
            x0 = res.x
            points = _make_symmetric_points(n_hull, n_inner, *x0)
            s, _ = compute_score(points)
            if s > best_score:
                best_score = s
                best_points = points.copy()
        except Exception:
            pass

    # Strategy 2: Fourier-based optimization
    if time.time() < start_time + time_limit - 30:
        for n_freq in [2, 3]:
            if time.time() > start_time + time_limit - 30:
                break
            
            x0 = np.zeros(n_freq * 3)
            x0[0] = 1.0
            x0[1] = 0.0
            x0[2] = 0.0
            
            for beta in [30.0, 80.0]:
                if time.time() > start_time + time_limit - 30:
                    break
                try:
                    result = opt.minimize(
                        lambda x, b=beta, nf=n_freq: _fourier_objective(x, nf, b),
                        x0,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    x0 = result.x
                    points = _fourier_points(x0.reshape(-1, 3))
                    s, _ = compute_score(points)
                    if s > best_score:
                        best_score = s
                        best_points = points.copy()
                except Exception:
                    pass

            if time.time() > start_time + time_limit - 30:
                break
            try:
                res = opt.minimize(
                    lambda x, nf=n_freq: _fourier_hard_objective(x, nf),
                    x0,
                    method='Powell',
                    options={'maxiter': 100, 'ftol': 1e-12, 'xtol': 1e-10}
                )
                x0 = res.x
                points = _fourier_points(x0.reshape(-1, 3))
                s, _ = compute_score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
            except Exception:
                pass

    # Strategy 3: Full free-space optimization from best result
    if best_points is not None and time.time() < start_time + time_limit - 20:
        current = best_points.copy()
        
        for beta in [30.0, 80.0]:
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

    # Strategy 4: Directed local search (STRUCTURAL CHANGE)
    if best_points is not None and time.time() < start_time + time_limit - 10:
        np.random.seed(42)
        dl_points, dl_score = _directed_local_search(best_points, max_iters=80, step_scale=0.005)
        if dl_score > best_score:
            best_score = dl_score
            best_points = dl_points.copy()

        # Second directed search with different parameters
        if time.time() < start_time + time_limit - 5:
            dl2_points, dl2_score = _directed_local_search(best_points, max_iters=60, step_scale=0.002)
            if dl2_score > best_score:
                best_score = dl2_score
                best_points = dl2_points.copy()

    # Strategy 5: Random restarts with directed search
    if time.time() < start_time + time_limit - 8:
        np.random.seed(123)
        for _ in range(3):
            if time.time() > start_time + time_limit - 8:
                break
            
            n_hull = np.random.randint(9, 13)
            n_inner = N - n_hull
            r_hull = np.random.uniform(0.9, 1.2)
            r_inner = np.random.uniform(0.2, 0.5) if n_inner > 0 else 0.0
            rot_hull = np.random.uniform(0, 2*np.pi)
            rot_inner = np.random.uniform(0, 2*np.pi) if n_inner > 1 else 0.0
            
            config = _make_symmetric_points(n_hull, n_inner, r_hull, r_inner, rot_hull, rot_inner)
            config = config + np.random.normal(0, 0.03, config.shape)

            s, _ = compute_score(config)
            if s > best_score:
                best_score = s
                best_points = config.copy()

            if time.time() > start_time + time_limit - 8:
                break

            # Quick directed search from this config
            dl_pts, dl_sc = _directed_local_search(config, max_iters=40, step_scale=0.008)
            if dl_sc > best_score:
                best_score = dl_sc
                best_points = dl_pts.copy()

    # Final polish with Powell on hard objective
    if best_points is not None and time.time() < start_time + time_limit - 3:
        try:
            res = opt.minimize(
                _hard_score_neg,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            final_points = res.x.reshape(-1, 2)
            final_score, _ = compute_score(final_points)
            if final_score > best_score:
                best_score = final_score
                best_points = final_points.copy()
        except Exception:
            pass

        # Final directed micro-search
        if time.time() < start_time + time_limit - 1:
            dl_final, dl_final_score = _directed_local_search(best_points, max_iters=30, step_scale=0.001)
            if dl_final_score > best_score:
                best_score = dl_final_score
                best_points = dl_final.copy()

    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

# EVOLVE-BLOCK-END
