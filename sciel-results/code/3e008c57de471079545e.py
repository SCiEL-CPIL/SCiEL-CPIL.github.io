"""Optimized program for Heilbronn convex regions: 13 points maximizing min triangle area / hull area.

Strategy:
- Parametrize points as a regular 13-gon with independent radial perturbations (13 DOF)
  to structurally enforce convexity and reduce the search space.
- Use a smooth log-sum-exp surrogate for the minimum triangle area.
- Optimize radial offsets via L-BFGS-B (gradient-based, fast, well-suited to smooth objectives).
- Also run a full 2D L-BFGS-B polish from multiple diverse initializations.
- Differential evolution in the 13-D radial space for global search.
- Final exact-score polish with Nelder-Mead on the best candidate.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 13
TIME_LIMIT = 140.0  # safety margin before 150s kill

# Precompute all triangle index triples for fast vectorized area computation
_TRIS = np.array(list(combinations(range(N), 3)))
_N_TRIS = _TRIS.shape[0]


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
    """Compute soft-min of areas using log-sum-exp trick.
    soft_min = -log(sum(exp(-beta * areas))) / beta
    """
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
    """Negative soft-min score for minimization."""
    pts = x.reshape(N, 2)
    return -_soft_min_score(pts, beta)


def _neg_score(x):
    """Negative exact score for minimization."""
    pts = x.reshape(N, 2)
    return -_score(pts)


def _radial_to_points(radii, base_angles=None):
    """Convert radial offsets (13 values) to 2D points on a regular 13-gon.
    base_angles are the angles of the regular polygon vertices.
    radii are the radial distances from center for each vertex.
    """
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([radii * np.cos(base_angles), radii * np.sin(base_angles)])
    return pts


def _radial_objective(radii, beta, base_angles=None):
    """Objective for optimizing radial offsets. Returns negative soft-min score."""
    if base_angles is None:
        base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = _radial_to_points(radii, base_angles)
    return -_soft_min_score(pts, beta)


def _regular_polygon(n=N, noise=0.0, rng=None):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    if noise > 0 and rng is not None:
        pts = pts + rng.normal(0, noise, (n, 2))
    return pts


def _center_ring(n=N, r_inner=0.5, r_outer=1.0, rng=None):
    """Points on concentric rings."""
    pts = np.zeros((n, 2))
    for i in range(n):
        angle = 2 * np.pi * i / n
        r = r_inner + (r_outer - r_inner) * (i % 2)
        pts[i, 0] = r * np.cos(angle)
        pts[i, 1] = r * np.sin(angle)
    if rng is not None:
        pts = pts + rng.normal(0, 0.05, (n, 2))
    return pts


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
    """Generate multiple initial configurations."""
    rng = np.random.default_rng(42)
    inits = []

    # 1. Regular 13-gon on unit circle
    inits.append(_regular_polygon())

    # 2. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2, 0.3, 0.5]:
        inits.append(_regular_polygon(noise=noise, rng=rng))

    # 3. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    inits.append(np.column_stack([x, y]))

    # 4. Center-ring configurations
    for r_inner in [0.3, 0.5, 0.7]:
        inits.append(_center_ring(r_inner=r_inner, rng=rng))

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


def _lbfgs_polish(pts, time_budget=15.0, start_time=None, betas=(50, 100, 200, 500)):
    """L-BFGS-B polish using soft-min surrogate with increasing beta."""
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
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
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


def _radial_optimization(start_time, time_budget=30.0):
    """Optimize radial offsets of a regular 13-gon using L-BFGS-B.
    This reduces the problem to 13 scalar parameters and enforces convexity."""
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    best_pts = None
    best_score = -1

    # Try multiple initial radial configurations
    rng = np.random.default_rng(7)
    radial_inits = [
        np.ones(N),  # regular polygon
        np.ones(N) + rng.normal(0, 0.1, N),
        np.ones(N) + rng.normal(0, 0.2, N),
        np.ones(N) + rng.normal(0, 0.3, N),
        np.ones(N) + rng.uniform(-0.2, 0.2, N),
        np.linspace(0.7, 1.3, N),
        np.ones(N) * rng.uniform(0.8, 1.2, N),
    ]

    for radii0 in radial_inits:
        if time.time() > deadline - 5:
            break

        # Optimize with increasing beta
        current_radii = radii0.copy()
        for beta in [20, 50, 100, 200, 500]:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(
                    _radial_objective,
                    current_radii,
                    args=(beta, base_angles),
                    method='L-BFGS-B',
                    bounds=[(0.1, 2.0)] * N,
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                current_radii = res.x
            except Exception:
                break

        pts = _radial_to_points(current_radii, base_angles)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Also try with a rotated base polygon
    for rot in [np.pi / 13, np.pi / 7]:
        if time.time() > deadline - 5:
            break
        rotated_angles = base_angles + rot
        for radii0 in [np.ones(N), np.ones(N) + rng.normal(0, 0.15, N)]:
            current_radii = radii0.copy()
            for beta in [50, 100, 300]:
                if time.time() > deadline - 3:
                    break
                try:
                    res = minimize(
                        _radial_objective,
                        current_radii,
                        args=(beta, rotated_angles),
                        method='L-BFGS-B',
                        bounds=[(0.1, 2.0)] * N,
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    current_radii = res.x
                except Exception:
                    break

            pts = _radial_to_points(current_radii, rotated_angles)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts.copy()

    if best_pts is None:
        return None, 0.0
    return best_pts, best_score


def _de_radial_search(start_time, time_budget=25.0):
    """Differential evolution in the 13-D radial space."""
    deadline = start_time + time_budget
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    def de_obj(radii):
        return _radial_objective(radii, beta=50, base_angles=base_angles)

    try:
        res = differential_evolution(
            de_obj,
            bounds=[(0.3, 1.8)] * N,
            seed=42,
            maxiter=60,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='latinhypercube',
            updating='immediate',
            callback=lambda xk, convergence: time.time() > deadline - 5
        )
        pts = _radial_to_points(res.x, base_angles)
        s = _score(pts)
        return pts, s
    except Exception:
        return None, 0.0


def _de_full_search(start_time, time_budget=25.0):
    """Differential evolution in the full 26-D space."""
    deadline = start_time + time_budget

    def de_obj(x):
        return _neg_soft_min_score(x, beta=50)

    try:
        res = differential_evolution(
            de_obj,
            bounds=[(-1.5, 1.5)] * (2 * N),
            seed=123,
            maxiter=50,
            popsize=12,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            init='halton',
            updating='immediate',
            callback=lambda xk, convergence: time.time() > deadline - 5
        )
        pts = res.x.reshape(N, 2)
        s = _score(pts)
        return pts, s
    except Exception:
        return None, 0.0


def _nelder_mead_polish(pts, time_budget=10.0, start_time=None):
    """Nelder-Mead optimization on the full 2N-dimensional space."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    x0 = pts.flatten().copy()
    try:
        result = minimize(
            _neg_score,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 1500,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        return result.x.reshape(N, 2), -result.fun
    except Exception:
        return pts, _score(pts)


def _targeted_polish(pts, time_budget=15.0, start_time=None):
    """Targeted polish: identify worst triangles and optimize points involved."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Find the worst (smallest area) triangles
    areas = _triangle_areas(best_pts)
    k_worst = min(5, len(areas))
    worst_indices = np.argsort(areas)[:k_worst]

    # Map triangle index to point indices
    def tri_idx_to_pts(tri_idx):
        count = 0
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                for k in range(j + 1, N):
                    if count == tri_idx:
                        return (i, j, k)
                    count += 1
        return (0, 1, 2)

    involved_points = set()
    for wi in worst_indices:
        i, j, k = tri_idx_to_pts(wi)
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
                options={
                    'maxiter': 400,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if -res.fun > best_score + 1e-12:
                best_pts[pt_idx] = res.x
                best_score = -res.fun
        except Exception:
            pass

    return best_pts, best_score


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    inits = _generate_initializations()

    best_pts = None
    best_score = -1

    # Evaluate all initializations
    for i, init in enumerate(inits):
        if time.time() > deadline - 50:
            break
        s = _score(init)
        if s > best_score:
            best_score = s
            best_pts = init.copy()

    if best_pts is None:
        best_pts = _regular_polygon()
        best_score = _score(best_pts)

    # Phase 1: Radial optimization (13-D, structurally convex)
    remaining = deadline - 30 - time.time()
    if remaining > 15:
        rad_pts, rad_score = _radial_optimization(start_time, time_budget=min(remaining, 25.0))
        if rad_pts is not None and rad_score > best_score:
            best_score = rad_score
            best_pts = rad_pts

    # Phase 2: L-BFGS-B polish on best candidates
    candidates_to_polish = [best_pts]
    # Add top initializations
    scored_inits = []
    for init in inits:
        s = _score(init)
        scored_inits.append((s, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    for s, init in scored_inits[:5]:
        if s > best_score * 0.6:
            candidates_to_polish.append(init)

    for cand in candidates_to_polish:
        if time.time() > deadline - 25:
            break
        remaining = deadline - 20 - time.time()
        if remaining < 5:
            break
        pts_opt, score_opt = _lbfgs_polish(
            cand,
            time_budget=min(remaining, 12.0),
            start_time=start_time,
            betas=(50, 100, 200, 500)
        )
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 3: Differential evolution (radial space)
    remaining = deadline - 20 - time.time()
    if remaining > 15:
        de_pts, de_score = _de_radial_search(start_time, time_budget=min(remaining, 20.0))
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts
        # Polish DE result
        if de_pts is not None and time.time() < deadline - 10:
            pts_opt, score_opt = _lbfgs_polish(
                de_pts,
                time_budget=min(deadline - 10 - time.time(), 8.0),
                start_time=start_time,
                betas=(100, 300, 500)
            )
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt

    # Phase 4: Differential evolution (full 2D space)
    remaining = deadline - 15 - time.time()
    if remaining > 15:
        de2_pts, de2_score = _de_full_search(start_time, time_budget=min(remaining, 18.0))
        if de2_pts is not None and de2_score > best_score:
            best_score = de2_score
            best_pts = de2_pts
        # Polish
        if de2_pts is not None and time.time() < deadline - 8:
            pts_opt, score_opt = _lbfgs_polish(
                de2_pts,
                time_budget=min(deadline - 8 - time.time(), 6.0),
                start_time=start_time,
                betas=(100, 300)
            )
            if score_opt > best_score:
                best_score = score_opt
                best_pts = pts_opt

    # Phase 5: Targeted polish on the best result
    remaining = deadline - 6 - time.time()
    if remaining > 5:
        best_pts, best_score = _targeted_polish(
            best_pts,
            time_budget=min(remaining, 8.0),
            start_time=start_time
        )

    # Phase 6: Final L-BFGS-B + Nelder-Mead polish
    remaining = deadline - 2 - time.time()
    if remaining > 3:
        best_pts, best_score = _lbfgs_polish(
            best_pts,
            time_budget=min(remaining * 0.6, 5.0),
            start_time=start_time,
            betas=(200, 500, 1000)
        )
        remaining2 = deadline - 1 - time.time()
        if remaining2 > 2:
            best_pts, best_score = _nelder_mead_polish(
                best_pts,
                time_budget=min(remaining2, 3.0),
                start_time=start_time
            )

    # Final safety check
    if time.time() > deadline - 1:
        return best_pts

    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score

    return best_pts
