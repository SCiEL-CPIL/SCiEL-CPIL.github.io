import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])
N_TRIPLES = TRIPLE_INDICES.shape[0]

# Precompute which triples involve each point for fast local updates
POINT_TRIPLES = [[] for _ in range(N)]
for idx, (i, j, k) in enumerate(TRIPLE_INDICES):
    POINT_TRIPLES[i].append(idx)
    POINT_TRIPLES[j].append(idx)
    POINT_TRIPLES[k].append(idx)
POINT_TRIPLES = [np.array(pt) for pt in POINT_TRIPLES]

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]

    min_area = 0.5 * np.min(np.abs(cross))
    return min_area / hull_area

def _compute_areas(points):
    """Compute all triangle areas (0.5 * |cross|) for a point set."""
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_value(areas, temp):
    """Soft minimum via log-sum-exp. Returns a smooth lower bound on min(areas)."""
    min_a = np.min(areas)
    if min_a < 1e-12:
        return 0.0
    # Use the standard soft-min: min(x) ≈ -temp * log(sum(exp(-x_i/temp)))
    # = min(x) - temp * log(sum(exp(-(x_i - min(x))/temp)))
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-shifted / temp)))
    return min_a - temp * log_sum

def _smooth_objective(x, temp=0.05):
    """Negative smooth score for L-BFGS-B optimization."""
    p = x.reshape(N, 2)
    if p.shape != (N, 2):
        return 1e6

    try:
        hull = ConvexHull(p)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6

    areas = _compute_areas(p)
    soft_min = _soft_min_value(areas, temp)

    return -soft_min / hull_area

def _generate_jittered_polygon(n, noise_amplitude, noise_seeds, affine_params):
    """
    Generate n points as a jittered regular n-gon.
    Each vertex is displaced radially by a small independent noise term.
    """
    angles = 2 * np.pi * np.arange(n) / n
    base_pts = np.column_stack((np.cos(angles), np.sin(angles)))

    radial_dirs = base_pts / np.linalg.norm(base_pts, axis=1, keepdims=True)
    jitter = noise_amplitude * noise_seeds[:, np.newaxis] * radial_dirs
    pts = base_pts + jitter

    scale, rotation, shear = affine_params
    R = np.array([[np.cos(rotation), -np.sin(rotation)],
                  [np.sin(rotation), np.cos(rotation)]])
    S = np.array([[1, shear], [0, 1]])
    A = scale * S @ R
    pts = pts @ A.T
    return pts

def _jittered_objective(params, noise_seeds):
    noise_amplitude = np.clip(params[0], 0.0, 0.5)
    affine_clipped = np.clip(params[1:4], [-2, -2, -2], [2, 2, 2])
    pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine_clipped)
    score = _calculate_score(pts)
    return -score

def _jittered_smooth_objective(params, noise_seeds, temp=0.03):
    noise_amplitude = np.clip(params[0], 0.0, 0.5)
    affine_clipped = np.clip(params[1:4], [-2, -2, -2], [2, 2, 2])
    pts = _generate_jittered_polygon(N, noise_amplitude, noise_seeds, affine_clipped)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 1e6
    except Exception:
        return 1e6
    areas = _compute_areas(pts)
    soft_min = _soft_min_value(areas, temp)
    return -soft_min / hull_area

def _optimize_jittered_polygon(time_limit=20.0):
    start_time = time.time()
    best_score = 0
    best_points = None

    rng = np.random.RandomState(42)
    n_restarts = 12

    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break

        if restart == 0:
            noise_seeds = np.zeros(N)
        elif restart == 1:
            noise_seeds = rng.randn(N) * 0.1
        elif restart == 2:
            noise_seeds = rng.randn(N) * 0.2
        else:
            noise_scale = rng.uniform(0.05, 0.3)
            noise_seeds = rng.randn(N) * noise_scale

        if restart == 0:
            x0 = np.array([0.0, 1.0, 0.0, 0.0])
        else:
            x0 = np.array([
                rng.uniform(0.0, 0.3),
                1.0 + rng.uniform(-0.2, 0.2),
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3)
            ])

        try:
            res = minimize(_jittered_objective, x0, args=(noise_seeds,), method='Nelder-Mead',
                          options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 500})
            pts = _generate_jittered_polygon(N, np.clip(res.x[0], 0.0, 0.5), noise_seeds,
                                             np.clip(res.x[1:4], [-2, -2, -2], [2, 2, 2]))
            s = _calculate_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

        if time.time() - start_time < time_limit * 0.7:
            try:
                res2 = minimize(_jittered_smooth_objective, x0, args=(noise_seeds,), method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                pts = _generate_jittered_polygon(N, np.clip(res2.x[0], 0.0, 0.5), noise_seeds,
                                                 np.clip(res2.x[1:4], [-2, -2, -2], [2, 2, 2]))
                s = _calculate_score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass

    return best_score, best_points

def _generate_two_ring(n_out, n_in, r_in, phi_rel):
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + phi_rel
    pts_out = np.column_stack((np.cos(angles_out), np.sin(angles_out)))
    if n_in > 0:
        pts_in = np.column_stack((r_in * np.cos(angles_in), r_in * np.sin(angles_in)))
        return np.vstack((pts_out, pts_in))
    else:
        return pts_out

def _optimize_two_ring(n_out, n_in, time_limit=8.0):
    start_time = time.time()

    def objective(params):
        r_in, phi_rel = params
        if r_in < 0 or r_in > 1.5:
            return 1e6
        pts = _generate_two_ring(n_out, n_in, r_in, phi_rel)
        score = _calculate_score(pts)
        return -score

    best_score = 0
    best_params = None

    for r0 in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        phi_candidates = [0.0]
        if n_in > 0:
            phi_candidates += [np.pi / n_in, np.pi / (2 * n_in)]
        if n_out > 0:
            phi_candidates.append(np.pi / n_out)

        for phi0 in phi_candidates:
            x0 = [r0, phi0]
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 300})
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_params = res.x.copy()
            except Exception:
                pass
            if time.time() - start_time > time_limit:
                return best_score, best_params

    return best_score, best_params

def _fast_coordinate_descent(points, time_limit=2.0):
    """
    Fast coordinate descent using precomputed POINT_TRIPLES for local area updates.
    Only recomputes areas for triples involving the moved point.
    """
    start_time = time.time()
    pts = points.copy()
    areas = _compute_areas(pts)
    min_area = np.min(areas)

    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

    for step in step_sizes:
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(N):
                tri_idx = POINT_TRIPLES[i]
                p_i_curr = pts[i]
                best_move = None
                best_min_area = min_area

                # Try 8 directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        new_p = p_i_curr + np.array([dx, dy])

                        # Only update areas for triples involving point i
                        new_areas = areas.copy()
                        for t_idx in tri_idx:
                            i_t, j_t, k_t = TRIPLE_INDICES[t_idx]
                            p_a = pts[i_t] if i_t != i else new_p
                            p_b = pts[j_t] if j_t != i else new_p
                            p_c = pts[k_t] if k_t != i else new_p
                            v1_t = p_b - p_a
                            v2_t = p_c - p_a
                            cross_t = v1_t[0] * v2_t[1] - v1_t[1] * v2_t[0]
                            new_areas[t_idx] = 0.5 * np.abs(cross_t)

                        curr_min = np.min(new_areas)
                        if curr_min > best_min_area:
                            best_min_area = curr_min
                            best_move = new_p

                if best_move is not None:
                    pts[i] = best_move
                    min_area = best_min_area
                    improved = True
                    # Recompute all areas for consistency
                    areas = _compute_areas(pts)
                    min_area = np.min(areas)

    return pts

def _polish_points(points, time_limit=5.0):
    """
    Multi-phase polish:
    Phase 1: L-BFGS-B with decreasing temperature soft-min
    Phase 2: Perturbation restarts with L-BFGS-B
    Phase 3: Fast coordinate descent
    """
    start_time = time.time()
    best_pts = points.copy()
    best_score = _calculate_score(points)

    # Phase 1: Smooth optimization with decreasing temperature
    temp_schedule = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

    for temp in temp_schedule:
        if time.time() - start_time > time_limit * 0.5:
            break

        def neg_smooth_score(x, t=temp):
            return _smooth_objective(x, t)

        try:
            res = minimize(neg_smooth_score, best_pts.flatten(), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
            new_pts = res.x.reshape(N, 2)
            new_score = _calculate_score(new_pts)
            if new_score > best_score:
                best_pts = new_pts
                best_score = new_score
        except Exception:
            pass

    # Phase 2: Perturbation-based restarts
    if time.time() - start_time < time_limit * 0.75:
        rng = np.random.RandomState(42)
        for n_restarts in range(5):
            if time.time() - start_time > time_limit * 0.75:
                break

            noise_scale = 0.005 * (n_restarts + 1)
            perturbed = best_pts + rng.randn(N, 2) * noise_scale

            def neg_smooth_score_pert(x, t=0.005):
                return _smooth_objective(x, t)

            try:
                res = minimize(neg_smooth_score_pert, perturbed.flatten(), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200})
                new_pts = res.x.reshape(N, 2)
                new_score = _calculate_score(new_pts)
                if new_score > best_score:
                    best_pts = new_pts
                    best_score = new_score
            except Exception:
                pass

    # Phase 3: Fast coordinate descent
    if time.time() - start_time < time_limit * 0.95:
        remaining = time_limit - (time.time() - start_time)
        best_pts = _fast_coordinate_descent(best_pts, max(1.0, remaining))
        best_score = _calculate_score(best_pts)

    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_score = 0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # Strategy 1: Jittered regular 14-gon optimization (primary approach)
    jitter_time = min(40.0, deadline - start_time - 40)
    jitter_score, jitter_pts = _optimize_jittered_polygon(time_limit=jitter_time)

    if jitter_pts is not None:
        try_update(jitter_score, jitter_pts)

        if time.time() < deadline - 20:
            remaining = deadline - time.time() - 15
            polished_pts, polished_score = _polish_points(jitter_pts, time_limit=min(25.0, remaining))
            try_update(polished_score, polished_pts)

    # Strategy 2: Two-ring optimization as backup/complement
    if time.time() < deadline - 30:
        splits = [
            (14, 0),
            (12, 2),
            (10, 4),
            (7, 7)
        ]

        for n_out, n_in in splits:
            if time.time() > deadline - 25:
                break

            if n_in == 0:
                pts = _generate_two_ring(n_out, 0, 0, 0)
                if time.time() < deadline - 20:
                    polished_pts, polished_score = _polish_points(pts, time_limit=5.0)
                    try_update(polished_score, polished_pts)
                continue

            remaining_for_ring = deadline - time.time() - 25
            if remaining_for_ring < 6:
                break
            ring_time = max(3.0, min(6.0, remaining_for_ring / 2.0))
            score, params = _optimize_two_ring(n_out, n_in, time_limit=ring_time)
            if params is not None:
                pts = _generate_two_ring(n_out, n_in, params[0], params[1])
                if time.time() < deadline - 20:
                    remaining = deadline - time.time() - 15
                    polish_time = max(2.0, min(5.0, remaining))
                    polished_pts, polished_score = _polish_points(pts, time_limit=polish_time)
                    try_update(polished_score, polished_pts)

    # Final intensive polish on best points
    if best_points is not None and time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        final_pts, final_score = _polish_points(best_points, time_limit=max(10.0, remaining))
        try_update(final_score, final_pts)

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        best_score = _calculate_score(best_points)
        remaining = deadline - time.time() - 3
        polished_pts, polished_score = _polish_points(best_points, time_limit=max(8.0, remaining))
        best_points = polished_pts
        best_score = polished_score

    return best_points
