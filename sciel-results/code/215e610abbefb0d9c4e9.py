"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Combines radial-parameterized 13-gon (structural convexity) with full 2D L-BFGS-B
polish, differential evolution in radial space, and targeted worst-triangle refinement.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# Precompute all triangle index triples
_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)


def _triangle_areas(pts):
    """Vectorized computation of all C(N,3) triangle areas."""
    i, j, k = _TRIS[:, 0], _TRIS[:, 1], _TRIS[:, 2]
    p1 = pts[i]
    p2 = pts[j]
    p3 = pts[k]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    """Compute min triangle area / convex hull area."""
    try:
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        min_area = np.min(_triangle_areas(pts))
        return min_area / hull_area
    except Exception:
        return 0.0


def _soft_min_value(areas, beta):
    """Compute soft-min of areas using log-sum-exp trick."""
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - log_sum / beta


def _soft_min_score(pts, beta):
    """Smooth surrogate: soft-min of triangle areas divided by hull area."""
    try:
        areas = _triangle_areas(pts)
        if len(areas) == 0:
            return 0.0
        hull_area = _hull_area(pts)
        if hull_area < 1e-12:
            return 0.0
        soft_min_val = _soft_min_value(areas, beta)
        return soft_min_val / hull_area
    except Exception:
        return 0.0


def _neg_soft_min_score(x, beta):
    """Negative soft-min score for minimization (full 2D)."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)


def _neg_score(x):
    """Negative exact score for minimization (full 2D)."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def _radial_to_points(radii, base_angles=None):
    """Convert radial offsets to 2D points on a regular N-gon."""
    if base_angles is None:
        base_angles = _BASE_ANGLES
    pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    return pts


def _radial_objective(radii, beta, base_angles=None):
    """Objective for optimizing radial offsets. Returns negative soft-min score."""
    if base_angles is None:
        base_angles = _BASE_ANGLES
    pts = _radial_to_points(radii, base_angles)
    return -_soft_min_score(pts, beta)


def _halton_sequence(n, base):
    """Generate n Halton sequence values in [0,1)."""
    result = []
    f = 1.0
    for i in range(1, n + 1):
        x = 0.0
        i_temp = i
        while i_temp > 0:
            f /= base
            x += f * (i_temp % base)
            i_temp //= base
        result.append(x)
    return np.array(result)


def _generate_initializations():
    """Generate multiple diverse initial configurations."""
    rng = np.random.default_rng(42)
    inits = []

    # 1. Regular 13-gon on unit circle
    inits.append(_radial_to_points(np.ones(N)))

    # 2. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2, 0.3, 0.5]:
        inits.append(_radial_to_points(np.ones(N) + rng.normal(0, noise, N)))

    # 3. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    inits.append(np.column_stack([x, y]))

    # 4. Center-ring configurations
    for r_inner in [0.3, 0.5, 0.7]:
        radii = np.ones(N)
        radii[::2] = r_inner
        inits.append(_radial_to_points(radii))

    # 5. Random points on unit circle (sorted angles)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(pts)

    # 6. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2 * np.pi, N)
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        inits.append(pts)

    # 7. Grid-like: mix of boundary and interior
    angles_boundary = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    boundary_pts = np.column_stack([np.cos(angles_boundary), np.sin(angles_boundary)])
    interior_angles = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi / 4
    interior_radii = [0.3, 0.5, 0.3, 0.5]
    interior_pts = np.column_stack([
        [r * np.cos(a) for r, a in zip(interior_radii, interior_angles)],
        [r * np.sin(a) for r, a in zip(interior_radii, interior_angles)]
    ])
    inits.append(np.vstack([boundary_pts, interior_pts]))

    # 8. Regular polygon with center point
    angles_12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle_12 = np.column_stack([np.cos(angles_12), np.sin(angles_12)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([circle_12, center]))

    return inits


def _radial_lbfgs_optimize(radii, beta, time_budget=10.0, start_time=None):
    """L-BFGS-B optimization in radial space (13 DOF)."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    try:
        res = minimize(
            _radial_objective,
            radii.copy(),
            args=(beta,),
            method='L-BFGS-B',
            bounds=[(0.1, 2.0)] * N,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        res_pts = _radial_to_points(res.x)
        return res_pts, _score(res_pts)
    except Exception:
        pts = _radial_to_points(radii)
        return pts, _score(pts)


def _radial_progressive_optimize(radii, start_time=None):
    """Progressive beta annealing in radial space."""
    if start_time is None:
        start_time = time.time()

    best_pts = _radial_to_points(radii)
    best_score = _score(best_pts)
    current_radii = radii.copy()

    betas = [10, 20, 40, 80, 150, 200, 300, 500]

    for beta in betas:
        if time.time() > start_time + TIME_LIMIT - 20:
            break
        remaining = TIME_LIMIT - 15 - (time.time() - start_time)
        if remaining < 5:
            break
        stage_time = min(remaining * 0.25, 12.0)

        pts_opt, score_opt = _radial_lbfgs_optimize(current_radii, beta,
                                                     time_budget=stage_time,
                                                     start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt
            # Recover radii from optimized points (project back)
            cos_a = np.cos(_BASE_ANGLES)
            sin_a = np.sin(_BASE_ANGLES)
            current_radii = best_pts[:, 0] * cos_a + best_pts[:, 1] * sin_a

    return best_pts, best_score


def _full_lbfgs_polish(pts, time_budget=15.0, start_time=None, betas=(50, 100, 200, 500)):
    """Full 2D L-BFGS-B polish using soft-min surrogate with increasing beta."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    x0 = pts.flatten().copy()
    best_pts = pts.copy()
    best_score = _score(best_pts)

    for beta in betas:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                _neg_soft_min_score,
                x0,
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
                x0 = res.x.copy()
        except Exception:
            break

    return best_pts, best_score


def _differential_evolution_radial(start_time, time_budget=20.0):
    """Differential evolution in 13-D radial space for global search."""
    deadline = start_time + time_budget
    if time.time() > deadline:
        return None, -1

    bounds = [(0.3, 1.8)] * N

    def de_objective(radii):
        pts = _radial_to_points(radii)
        return -_score(pts)

    try:
        de_result = differential_evolution(
            de_objective,
            bounds=bounds,
            seed=42,
            maxiter=80,
            tol=1e-8,
            popsize=15,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='latinhypercube'
        )
        if time.time() > deadline:
            return None, -1
        best_radii = de_result.x
        best_pts = _radial_to_points(best_radii)
        best_score = _score(best_pts)
        return best_pts, best_score
    except Exception:
        return None, -1


def _targeted_polish(pts, time_budget=15.0, start_time=None):
    """Targeted polish: identify worst triangles and optimize points involved."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Find the worst (smallest area) triangles
    areas = _triangle_areas(best_pts)
    n_triangles = len(areas)
    k_worst = min(5, n_triangles)
    worst_indices = np.argsort(areas)[:k_worst]

    involved_points = set()
    for wi in worst_indices:
        i, j, k = _TRIS[wi]
        involved_points.add(i)
        involved_points.add(j)
        involved_points.add(k)

    # Optimize each involved point using Nelder-Mead on just that point's coordinates
    for pt_idx in sorted(involved_points):
        if time.time() > deadline - 2:
            break

        fixed_pts = best_pts.copy()
        x0 = fixed_pts[pt_idx].copy()

        def obj_single(x, idx=pt_idx, fp=fixed_pts):
            trial = fp.copy()
            trial[idx] = x
            return -_score(trial)

        try:
            res = minimize(
                obj_single,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
        except Exception:
            continue

        if time.time() > deadline - 1:
            break

        if -res.fun > best_score + 1e-12:
            best_pts[pt_idx] = res.x
            best_score = -res.fun

    return best_pts, best_score


def _nelder_mead_polish(pts, time_budget=10.0, start_time=None):
    """Nelder-Mead optimization on the full 2N-dimensional space for final polish."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    x0 = pts.flatten().copy()

    try:
        result = minimize(
            _neg_score,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
    except Exception:
        return pts, _score(pts)

    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return pts, _score(pts)

    return result.x.reshape(N, 2), -result.fun


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    inits = _generate_initializations()

    best_pts = None
    best_score = -1

    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 60:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()

    if best_pts is None:
        best_pts = _radial_to_points(np.ones(N))
        best_score = _score(best_pts)

    # Phase 1: Radial-space optimization from multiple initializations
    # This is the key structural change: 13 DOF, convexity enforced
    for i, init in enumerate(inits):
        if time.time() > deadline - 40:
            break
        s_init = _score(init)
        if s_init < best_score * 0.5:
            continue

        # Project init onto radial representation
        cos_a = np.cos(_BASE_ANGLES)
        sin_a = np.sin(_BASE_ANGLES)
        radii = init[:, 0] * cos_a + init[:, 1] * sin_a
        radii = np.clip(radii, 0.1, 2.0)

        remaining = deadline - 30 - time.time()
        if remaining < 8:
            break

        pts_opt, score_opt = _radial_progressive_optimize(radii, start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 2: Differential evolution in radial space for global search
    remaining = deadline - 30 - time.time()
    if remaining > 15:
        de_pts, de_score = _differential_evolution_radial(start_time, time_budget=min(remaining, 25.0))
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts

    # Phase 3: Full 2D L-BFGS-B polish on best candidate
    remaining = deadline - 20 - time.time()
    if remaining > 8:
        pts_opt, score_opt = _full_lbfgs_polish(best_pts, time_budget=min(remaining, 15.0),
                                                 start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 4: Targeted polish on worst triangles
    remaining = deadline - 10 - time.time()
    if remaining > 5:
        best_pts, best_score = _targeted_polish(best_pts, time_budget=min(remaining, 12.0),
                                                 start_time=start_time)

    # Phase 5: Final L-BFGS-B with high beta in full 2D
    remaining = deadline - 5 - time.time()
    if remaining > 3:
        pts_opt, score_opt = _full_lbfgs_polish(best_pts, time_budget=min(remaining, 5.0),
                                                 start_time=start_time, betas=(300, 500))
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 6: Final Nelder-Mead polish for precision
    remaining = deadline - 2 - time.time()
    if remaining > 2:
        best_pts, best_score = _nelder_mead_polish(best_pts, time_budget=min(remaining, 3.0),
                                                    start_time=start_time)

    return best_pts
