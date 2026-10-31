import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
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


def _neg_score(points):
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: use log-sum-exp to approximate min of triangle areas.
    soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    Then score = soft_min / hull_area
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    # Clip to avoid numerical issues
    areas_clipped = np.clip(areas, 1e-15, None)
    # Log-sum-exp for soft min
    neg_beta_areas = -beta * areas_clipped
    max_val = np.max(neg_beta_areas)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min = -log_sum / beta
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, beta=50.0):
    s = _soft_min_score(points, beta)
    if s <= 0:
        return 1e10
    return -s


def _build_two_ring(n_out, r_inner, angle_offset, radial_pert=None, angular_pert=None):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)

    r_out = np.ones(n_out)
    r_in = np.full(n_in, r_inner)

    if radial_pert is not None:
        rp_out, rp_in = radial_pert[:n_out], radial_pert[n_out:]
        r_out = r_out * (1.0 + rp_out)
        r_in = r_in * (1.0 + rp_in)
    if angular_pert is not None:
        ap_out, ap_in = angular_pert[:n_out], angular_pert[n_out:]
        a_out = a_out + ap_out
        a_in = a_in + ap_in

    outer = np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])
    inner = np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)])
    return np.vstack([outer, inner])


def _build_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _build_regular_perturbed(base, radial_pert, angular_pert):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + angular_pert
    radii = 1.0 + radial_pert
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _build_symmetric_rings(n_outer, n_mid, n_inner, r_mid, r_inner, off_mid, off_inner):
    """Build a 3-ring configuration with specified counts."""
    a_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    a_mid = np.linspace(off_mid, 2 * np.pi + off_mid, n_mid, endpoint=False)
    a_in = np.linspace(off_inner, 2 * np.pi + off_inner, n_inner, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a_out), np.sin(a_out)]),
        np.column_stack([r_mid * np.cos(a_mid), r_mid * np.sin(a_mid)]),
        np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)]),
    ])
    return pts


def _optimize_smooth(points, deadline, beta=50.0, max_iters=500):
    """Optimize using smooth surrogate with L-BFGS-B, then refine with true objective."""
    best = points.copy()
    best_s = _score(best)

    x0 = best.flatten()

    # Phase 1: Smooth optimization with L-BFGS-B
    if time.time() < deadline - 15:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), beta),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    # Phase 2: Powell on true objective
    if time.time() < deadline - 10:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    # Phase 3: Nelder-Mead fine polish on true objective
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def _de_global_search(deadline, bounds=None, maxiter=80, popsize=20, seed=42):
    """Run differential evolution for global search."""
    if bounds is None:
        bounds = [(-1.5, 1.5)] * (2 * N)

    def de_objective(x):
        pts = x.reshape(N, 2)
        return _neg_score(pts)

    try:
        res = differential_evolution(
            de_objective,
            bounds=bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-7,
            seed=seed,
            polish=True,
            init='sobol',
            update_interval=1,
            disp=False,
        )
        return res.x.reshape(N, 2)
    except Exception:
        return None


def _optimize_two_ring(n_out, r_inner, angle_offset, deadline):
    """Optimize a two-ring model with per-point perturbations."""
    n_in = N - n_out
    p0 = np.zeros(2 + 2 * N)
    p0[0] = r_inner
    p0[1] = angle_offset

    def obj(p):
        r_in = float(p[0])
        a_off = float(p[1])
        rp = p[2:2 + N]
        ap = p[2 + N:2 + 2 * N]
        pts = _build_two_ring(n_out, r_in, a_off, radial_pert=rp, angular_pert=ap)
        return _neg_score(pts)

    x0 = p0
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 8:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 4:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_two_ring(n_out, float(best_p[0]), float(best_p[1]),
                          radial_pert=best_p[2:2 + N], angular_pert=best_p[2 + N:2 + 2 * N])
    return pts, best_s


def _optimize_three_ring(n1, n2, n3, r2, r3, off2, off3, deadline):
    """Optimize a three-ring model."""
    def obj(p):
        rr2, rr3, o2, o3 = p[0], p[1], p[2], p[3]
        pts = _build_three_ring(n1, n2, n3, rr2, rr3, o2, o3)
        return _neg_score(pts)

    x0 = np.array([r2, r3, off2, off3], dtype=float)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 6:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 300, 'xatol': 1e-9, 'fatol': 1e-9})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_three_ring(n1, n2, n3, float(best_p[0]), float(best_p[1]),
                            float(best_p[2]), float(best_p[3]))
    return pts, best_s


def _optimize_regular_perturbed(deadline):
    """Regular 13-gon with per-point radial/angular perturbations."""
    def obj(p):
        rp = p[:N]
        ap = p[N:2 * N]
        pts = _build_regular_perturbed(None, rp, ap)
        return _neg_score(pts)

    x0 = np.zeros(2 * N)
    best_p = x0.copy()
    best_s = -obj(x0)

    if time.time() < deadline - 8:
        try:
            res = minimize(obj, x0, method='Powell',
                           options={'maxiter': 400, 'xatol': 1e-9, 'fatol': 1e-9})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 4:
        try:
            res = minimize(obj, best_p, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if -res.fun > best_s:
                best_s, best_p = -res.fun, res.x.copy()
        except Exception:
            pass

    pts = _build_regular_perturbed(None, best_p[:N], best_p[N:2 * N])
    return pts, best_s


def _multi_beta_smooth_optimize(points, deadline):
    """Try multiple beta values for the soft-min surrogate to find better basin."""
    best = points.copy()
    best_s = _score(best)

    for beta in [20.0, 50.0, 100.0]:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), beta),
                best.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 1500},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s, best = s, cand
        except Exception:
            pass

    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s kill

    best_points = None
    best_score = 0.0

    def _update(pts):
        nonlocal best_points, best_score
        if pts is None or len(pts) != N:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # ---------- Seed candidates ----------
    # Regular 13-gon
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    _update(np.column_stack([np.cos(ang), np.sin(ang)]))

    # Two-ring: 7 outer, 6 inner
    for r_in in [0.5, 0.6, 0.7, 0.8]:
        for off in [0.0, np.pi / 6, np.pi / 3]:
            pts = _build_two_ring(7, r_in, off)
            _update(pts)

    # Three-ring: 6-4-3
    for r2 in [0.5, 0.65]:
        for r3 in [0.3, 0.4]:
            for o2 in [0.0, np.pi / 6]:
                for o3 in [0.0, np.pi / 4]:
                    pts = _build_three_ring(6, 4, 3, r2, r3, o2, o3)
                    _update(pts)

    # ---------- Phase 1: Smooth optimization from best seed ----------
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _multi_beta_smooth_optimize(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Differential Evolution global search ----------
    if time.time() < deadline - 50:
        try:
            de_pts = _de_global_search(deadline, maxiter=100, popsize=25, seed=42)
            if de_pts is not None:
                _update(de_pts)
        except Exception:
            pass

    # ---------- Phase 3: Smooth + local polish from DE result ----------
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _optimize_smooth(best_points, deadline, beta=50.0, max_iters=400)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 4: Model-based optimizations ----------
    # Two-ring: 7 outer, 6 inner
    if time.time() < deadline - 15:
        try:
            pts, s = _optimize_two_ring(7, 0.6, np.pi / 6, deadline)
            _update(pts)
        except Exception:
            pass

    # Two-ring: 8 outer, 5 inner
    if time.time() < deadline - 12:
        try:
            pts, s = _optimize_two_ring(8, 0.5, np.pi / 8, deadline)
            _update(pts)
        except Exception:
            pass

    # Three-ring: 6-4-3
    if time.time() < deadline - 10:
        try:
            pts, s = _optimize_three_ring(6, 4, 3, 0.6, 0.35, np.pi / 6, np.pi / 4, deadline)
            _update(pts)
        except Exception:
            pass

    # Regular perturbed
    if time.time() < deadline - 8:
        try:
            pts, s = _optimize_regular_perturbed(deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 5: Second DE with tighter bounds ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            center = best_points.mean(axis=0)
            spread = np.max(np.abs(best_points - center))
            lo = center - spread * 1.3
            hi = center + spread * 1.3
            bounds = [(lo[i % 2], hi[i % 2]) for i in range(2 * N)]

            def de_obj2(x):
                return _neg_score(x.reshape(N, 2))

            res = differential_evolution(
                de_obj2,
                bounds=bounds,
                maxiter=60,
                popsize=15,
                tol=1e-8,
                seed=123,
                polish=True,
                init='sobol',
                disp=False,
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 6: Final ultra-fine polish ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 7: One more smooth pass with very high beta ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_soft_score(x.reshape(N, 2), 200.0),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    if best_points is None:
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(ang), np.sin(ang)])

    return best_points
