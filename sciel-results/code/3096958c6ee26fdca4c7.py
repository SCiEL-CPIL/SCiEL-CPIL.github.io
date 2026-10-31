import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _soft_min_value(values, temperature):
    """Smooth min via temperature-based log-sum-exp (numerically stable)."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, 0, 700 * temperature)
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


# ---------- Support Line Parameterization ----------

def _support_lines_to_points(params):
    """
    Convert support line parameters to polygon vertices.
    params shape: (2N,) -> [theta_0..theta_{N-1}, offset_0..offset_{N-1}]
    
    Each line is n_i . x = offset_i, where n_i = (cos(theta_i), sin(theta_i)).
    The polygon is the intersection of all half-planes n_i . x <= offset_i.
    Vertices are intersections of adjacent lines (in angular order).
    """
    thetas = params[:N]
    offsets = params[N:]
    
    # Sort by angle to ensure convex ordering
    sort_idx = np.argsort(thetas)
    thetas_sorted = thetas[sort_idx]
    offsets_sorted = offsets[sort_idx]
    
    # Make angles cyclic
    thetas_sorted = thetas_sorted - thetas_sorted[0]
    
    points = np.zeros((N, 2))
    for i in range(N):
        # Line i: n_i . x = offset_i
        # Line (i+1)%N: n_{i+1} . x = offset_{i+1}
        i1 = (i + 1) % N
        
        n_i = np.array([np.cos(thetas_sorted[i]), np.sin(thetas_sorted[i])])
        n_j = np.array([np.cos(thetas_sorted[i1]), np.sin(thetas_sorted[i1])])
        
        # Solve: n_i . p = offset_i, n_j . p = offset_j
        A = np.array([n_i, n_j])
        b = np.array([offsets_sorted[i], offsets_sorted[i1]])
        
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            # Parallel lines, use midpoint as fallback
            points[i] = 0.5 * (offsets_sorted[i] * n_i + offsets_sorted[i1] * n_j)
        else:
            points[i] = np.linalg.solve(A, b)
    
    # Center the polygon at origin
    points = points - np.mean(points, axis=0)
    
    # Verify convexity: check that all points are inside all half-planes
    # If not convex, return a regular polygon as fallback
    for i in range(N):
        n_i = np.array([np.cos(thetas_sorted[i]), np.sin(thetas_sorted[i])])
        # Check all points satisfy n_i . x <= offset_i (with tolerance)
        dots = points @ n_i
        if np.any(dots > offsets_sorted[i] + 1e-6):
            # Not convex, fallback to regular polygon
            ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.column_stack([np.cos(ang), np.sin(ang)])
    
    return points


def _neg_soft_score_support(params, temperature):
    """Objective for support line optimization."""
    points = _support_lines_to_points(params)
    return _neg_soft_score(points, temperature)


def _neg_score_support(params):
    """Objective for support line optimization (hard min)."""
    points = _support_lines_to_points(params)
    return _neg_score(points)


def _regular_polygon_support_params():
    """Generate support line parameters for a regular 13-gon."""
    thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    # For a regular polygon centered at origin with circumradius R,
    # the inradius is R * cos(pi/N)
    R = 1.0
    offsets = R * np.cos(np.pi / N) * np.ones(N)
    return np.concatenate([thetas, offsets])


def _optimize_support_lines(params0, deadline, temp_start=0.05, temp_end=1e-5, n_steps=6):
    """Annealed soft-min optimization in support line parameter space."""
    best_params = params0.copy()
    points = _support_lines_to_points(best_params)
    best_s = _score(points)
    
    temps = np.exp(np.linspace(np.log(temp_start), np.log(temp_end), n_steps))
    
    cur = params0.copy()
    for temp in temps:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                lambda x, t=temp: _neg_soft_score_support(x, t),
                cur,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxfun': 1200},
            )
            cand_params = res.x
            cand_pts = _support_lines_to_points(cand_params)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best_params = cand_params.copy()
            cur = cand_params
        except Exception:
            pass
    
    # Final polish in Cartesian space
    points = _support_lines_to_points(best_params)
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                points = cand
        except Exception:
            pass
    
    return points, best_s


# ---------- Fourier Parameterization (from current program) ----------

def _fourier_to_polar(params):
    base_angle = params[0]
    a1, b1, a2, b2 = params[1], params[2], params[3], params[4]
    base_radius = params[5]
    r1 = params[6]
    
    i = np.arange(N)
    base_ang = 2 * np.pi * i / N
    angles = base_angle + base_ang + a1 * np.cos(base_ang) + b1 * np.sin(base_ang) + \
             a2 * np.cos(2 * base_ang) + b2 * np.sin(2 * base_ang)
    radii = base_radius + r1 * np.cos(base_ang)
    
    return np.concatenate([angles, radii])


def _polar_to_cartesian(polar):
    angles = polar[:N]
    radii = polar[N:]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])


def _fourier_objective(params):
    polar = _fourier_to_polar(params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _fourier_soft_objective(params, temperature):
    polar = _fourier_to_polar(params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


def _optimize_fourier(params0, deadline, temp_start=0.05, temp_end=1e-5,
                      max_total_iters=800, stages=4):
    best_params = params0.copy()
    best_s = -_fourier_objective(best_params)
    
    temps = np.exp(np.linspace(np.log(temp_start), np.log(temp_end), stages))
    iters_per_stage = max(50, max_total_iters // stages)
    
    cur = params0.copy()
    for temp in temps:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, t=temp: _fourier_soft_objective(x, t),
                cur,
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand_params = res.x
            s = -_fourier_objective(cand_params)
            if s > best_s:
                best_s = s
                best_params = cand_params.copy()
            cur = cand_params
        except Exception:
            pass
    
    polar = _fourier_to_polar(best_params)
    points = _polar_to_cartesian(polar)
    points = points - np.mean(points, axis=0)
    s = _score(points)
    if s > best_s:
        best_s = s
    
    if time.time() < deadline - 4:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12,
                         'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                points = cand
        except Exception:
            pass
    
    return points, best_s


# ---------- Annealed Soft-Min in Cartesian Space ----------

def _annealed_soft_minimize(points, deadline, temp_start=0.05, temp_end=1e-5,
                            max_total_iters=1200, stages=4):
    best = points.copy()
    best_s = _score(best)
    temps = np.exp(np.linspace(np.log(temp_start), np.log(temp_end), stages))
    iters_per_stage = max(50, max_total_iters // stages)

    cur = points.copy()
    for temp in temps:
        if time.time() > deadline - 3:
            break
        try:
            res = minimize(
                lambda x, t=temp: _neg_soft_score(x.reshape(N, 2), t),
                cur.flatten(),
                method='L-BFGS-B',
                options={'maxiter': iters_per_stage, 'ftol': 1e-12,
                         'gtol': 1e-10, 'maxfun': iters_per_stage * 4},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand.copy()
            cur = cand
        except Exception:
            pass

    if time.time() > deadline - 4:
        return best, best_s
    try:
        res = minimize(
            lambda x: -_score(x.reshape(N, 2)),
            best.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12,
                     'adaptive': True},
        )
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        if s > best_s:
            best_s, best = s, cand
    except Exception:
        pass
    return best, best_s


# ---------- Seed Generators ----------

def _seed_regular(radius=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def _seed_two_ring(n_out, r_in, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_quasi_regular(radial_pert, angular_pert, radius=1.0):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    rad = radius * (1.0 + radial_pert)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


# ---------- Simulated Annealing ----------

def _simulated_annealing(points, deadline, max_iters=5000, T_start=0.1, T_end=0.001):
    rng = np.random.default_rng(42)
    best = points.copy()
    best_s = _score(best)
    cur = points.copy()
    cur_s = best_s
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        T = T_start * (T_end / T_start) ** (it / max_iters)
        
        n_perturb = rng.integers(1, 3)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        step_size = T * 0.5
        new_pts = cur.copy()
        for idx in idxs:
            new_pts[idx] += rng.normal(0, step_size, size=2)
        
        new_s = _score(new_pts)
        if new_s >= cur_s or rng.random() < np.exp((new_s - cur_s) / max(T, 1e-10)):
            cur = new_pts
            cur_s = new_s
            if cur_s > best_s:
                best_s = cur_s
                best = cur.copy()
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 143.0

    best_points = _seed_regular()
    best_score = _score(best_points)

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Phase 1: Support line optimization (NEW structural approach)
    # This directly parameterizes the convex polygon via its supporting lines
    rng = np.random.default_rng(12345)
    n_support_starts = 8
    for start_idx in range(n_support_starts):
        if time.time() > deadline - 60:
            break
        
        if start_idx == 0:
            params0 = _regular_polygon_support_params()
        else:
            # Perturb the regular polygon's support lines
            base_params = _regular_polygon_support_params()
            thetas = base_params[:N] + rng.normal(0, 0.05, N)
            offsets = base_params[N:] * (1.0 + rng.normal(0, 0.1, N))
            params0 = np.concatenate([thetas, offsets])
        
        per_seed_budget = deadline - 60 - time.time()
        try:
            pts, s = _optimize_support_lines(params0, deadline,
                                             temp_start=0.05, temp_end=1e-5,
                                             n_steps=5)
            _update(pts)
        except Exception:
            pass

    # Phase 2: Fourier-space optimization
    params0_fourier = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    try:
        pts, s = _optimize_fourier(params0_fourier, deadline,
                                   temp_start=0.05, temp_end=1e-5,
                                   max_total_iters=600, stages=3)
        _update(pts)
    except Exception:
        pass

    # Phase 3: Cartesian-space optimization from multiple seeds
    seeds = []
    seeds.append(_seed_regular())
    seeds.append(_seed_regular(0.95))
    seeds.append(_seed_regular(1.05))

    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for n_out in [6, 7, 8]:
            for off in [0.0, np.pi / 7, np.pi / 6, np.pi / 5]:
                seeds.append(_seed_two_ring(n_out, r_in, off))

    rng = np.random.default_rng(7)
    for _ in range(20):
        rp = rng.uniform(-0.15, 0.15, N)
        ap = rng.uniform(-0.2, 0.2, N)
        seeds.append(_seed_quasi_regular(rp, ap))

    for _ in range(15):
        r = rng.uniform(0.3, 1.0, N)
        a = rng.uniform(0, 2 * np.pi, N)
        seeds.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))

    scored = []
    for s in seeds:
        sc = _score(s)
        scored.append((sc, s))
    scored.sort(key=lambda x: -x[0])
    top_seeds = [s for sc, s in scored[:12]]
    if _score(best_points) > scored[0][0]:
        top_seeds.insert(0, best_points.copy())

    for idx, seed in enumerate(top_seeds):
        if time.time() > deadline - 30:
            break
        per_seed_budget = deadline - 30 - time.time()
        max_iters = min(800, max(200, int(per_seed_budget * 15)))
        try:
            pts, s = _annealed_soft_minimize(seed, deadline,
                                             temp_start=0.05, temp_end=1e-5,
                                             max_total_iters=max_iters, stages=4)
            _update(pts)
        except Exception:
            pass

    # Phase 4: Simulated Annealing for global exploration
    if time.time() < deadline - 25:
        try:
            pts, s = _simulated_annealing(best_points, deadline, max_iters=3000)
            _update(pts)
            if time.time() < deadline - 15:
                pts2, s2 = _annealed_soft_minimize(pts, deadline,
                                                   temp_start=0.03, temp_end=1e-5,
                                                   max_total_iters=600, stages=3)
                _update(pts2)
        except Exception:
            pass

    # Phase 5: Final Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14,
                         'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 6: High-precision soft pass
    if time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 1e-5),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                         'maxfun': 800},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
