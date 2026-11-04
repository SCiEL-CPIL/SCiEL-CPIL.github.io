"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area.
Uses angular-ordering representation (points sorted by angle around centroid)
with simulated annealing on the true objective, coordinate descent, and
worst-triangle targeted repair.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
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
    inits.append(np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)]))

    # 2. Perturbed regular polygons
    for noise in [0.05, 0.1, 0.2, 0.3, 0.5]:
        radii = np.ones(N) + rng.normal(0, noise, N)
        inits.append(np.column_stack([radii * np.cos(_BASE_ANGLES), radii * np.sin(_BASE_ANGLES)]))

    # 3. Halton sequence in unit square
    x = _halton_sequence(N, 2)
    y = _halton_sequence(N, 3)
    inits.append(np.column_stack([x, y]))

    # 4. Center-ring configurations
    for r_inner in [0.3, 0.5, 0.7]:
        radii = np.ones(N)
        radii[::2] = r_inner
        inits.append(np.column_stack([radii * np.cos(_BASE_ANGLES), radii * np.sin(_BASE_ANGLES)]))

    # 5. Random points on unit circle (sorted angles)
    for _ in range(5):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        inits.append(pts)

    # 6. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = np.sort(rng.uniform(0, 2 * np.pi, N))
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


def _coordinate_descent_polish(pts, time_budget=15.0, start_time=None):
    """Coordinate descent: optimize each point's coordinates one at a time."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Iterate through points multiple times
    max_rounds = 50
    for round_num in range(max_rounds):
        if time.time() > deadline - 2:
            break
        improved = False
        for pt_idx in range(N):
            if time.time() > deadline - 1:
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
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                continue

            if -res.fun > best_score + 1e-12:
                best_pts[pt_idx] = res.x
                best_score = -res.fun
                improved = True

        if not improved:
            break

    return best_pts, best_score


def _simulated_annealing_search(pts, time_budget=30.0, start_time=None, seed=42):
    """Simulated annealing on the true objective (min triangle area / hull area)."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    rng = np.random.default_rng(seed)

    best_pts = pts.copy()
    best_score = _score(best_pts)
    current_pts = pts.copy()
    current_score = _score(current_pts)

    # Temperature schedule
    T_initial = 0.05
    T_final = 0.001
    n_iterations = 5000
    cooling_rate = (T_final / T_initial) ** (1.0 / n_iterations)

    T = T_initial

    for it in range(n_iterations):
        if time.time() > deadline - 2:
            break

        # Propose a random perturbation
        proposal = current_pts.copy()
        n_perturb = rng.integers(1, 4)  # Perturb 1-3 random points
        indices = rng.choice(N, size=n_perturb, replace=False)
        for idx in indices:
            proposal[idx] += rng.normal(0, T, size=2)

        # Evaluate
        proposal_score = _score(proposal)

        # Accept if better, or with probability exp((new-old)/T)
        if proposal_score > current_score:
            current_pts = proposal
            current_score = proposal_score
            if current_score > best_score:
                best_score = current_score
                best_pts = proposal.copy()
        else:
            delta = proposal_score - current_score
            if delta > 0:
                accept_prob = np.exp(delta / T)
            else:
                accept_prob = np.exp(delta / T) if T > 1e-10 else 0.0
            if rng.random() < accept_prob:
                current_pts = proposal
                current_score = proposal_score

        T *= cooling_rate

    return best_pts, best_score


def _targeted_repair(pts, time_budget=15.0, start_time=None):
    """Targeted repair: identify worst triangles and optimize points involved."""
    if start_time is None:
        start_time = time.time()
    deadline = start_time + time_budget

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Find the worst (smallest area) triangles
    areas = _triangle_areas(best_pts)
    n_triangles = len(areas)
    k_worst = min(8, n_triangles)
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
        best_pts = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])
        best_score = _score(best_pts)

    # Phase 1: Simulated annealing from best initialization
    remaining = deadline - 40 - time.time()
    if remaining > 10:
        pts_opt, score_opt = _simulated_annealing_search(best_pts, time_budget=min(remaining, 30.0),
                                                          start_time=start_time, seed=42)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 2: Coordinate descent polish
    remaining = deadline - 25 - time.time()
    if remaining > 8:
        pts_opt, score_opt = _coordinate_descent_polish(best_pts, time_budget=min(remaining, 15.0),
                                                         start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 3: Targeted repair on worst triangles
    remaining = deadline - 15 - time.time()
    if remaining > 5:
        best_pts, best_score = _targeted_repair(best_pts, time_budget=min(remaining, 12.0),
                                                 start_time=start_time)

    # Phase 4: Full 2D L-BFGS-B polish on best candidate
    remaining = deadline - 8 - time.time()
    if remaining > 5:
        pts_opt, score_opt = _full_lbfgs_polish(best_pts, time_budget=min(remaining, 10.0),
                                                 start_time=start_time)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 5: Another round of simulated annealing with different seed
    remaining = deadline - 5 - time.time()
    if remaining > 5:
        pts_opt, score_opt = _simulated_annealing_search(best_pts, time_budget=min(remaining, 8.0),
                                                          start_time=start_time, seed=123)
        if score_opt > best_score:
            best_score = score_opt
            best_pts = pts_opt

    # Phase 6: Final Nelder-Mead polish for precision
    remaining = deadline - 2 - time.time()
    if remaining > 2:
        best_pts, best_score = _nelder_mead_polish(best_pts, time_budget=min(remaining, 3.0),
                                                    start_time=start_time)

    return best_pts
