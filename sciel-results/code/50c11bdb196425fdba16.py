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


def _soft_min_value(values, temperature):
    """Smooth min via temperature-based log-sum-exp (numerically stable)."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
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


# ---------- Penalty-based formulation: maximize t s.t. area_i >= t ----------

def _penalty_objective(x, points, mu):
    """
    Penalty objective: minimize -t + mu * sum(max(0, t - area_i)^2)
    where x = [t, x1, y1, ..., x13, y13]
    """
    t = x[0]
    pts = x[1:].reshape(N, 2)
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    # We want to maximize t/hull_area, so minimize -t/hull_area + penalty
    return -t / hull_a + mu * penalty


def _penalty_optimize(points, deadline, mu_start=10.0, mu_end=1e6, n_mu_steps=8,
                      max_iter_per_mu=100):
    """
    Penalty-based interior-point: iteratively tighten mu.
    x = [t, x1, y1, ..., xN, yN]
    """
    best = points.copy()
    best_s = _score(best)
    
    # Initialize t to current min area
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.9
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:] = best.flatten()
    
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                lambda x: _penalty_objective(x, None, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-14, 'gtol': 1e-12},
            )
            x0 = res.x
            cand_pts = x0[1:].reshape(N, 2)
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s


# ---------- Radial/angular parameterization ----------

def _radial_to_cartesian(params):
    """Convert [r1..rN, theta1..thetaN] to Cartesian. Sort by angle."""
    radii = params[:N]
    angles = params[N:]
    # Sort by angle to maintain convex ordering
    order = np.argsort(angles)
    radii_sorted = radii[order]
    angles_sorted = angles[order]
    # Make angles cyclic (add 2*pi to wrap)
    angles_sorted = angles_sorted - angles_sorted[0]
    angles_sorted = np.sort(angles_sorted)
    pts = np.column_stack([radii_sorted * np.cos(angles_sorted),
                           radii_sorted * np.sin(angles_sorted)])
    return pts


def _cartesian_to_radial(points):
    """Convert Cartesian to [r1..rN, theta1..thetaN] sorted by angle."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    order = np.argsort(angles)
    return np.concatenate([radii[order], angles[order]])


def _neg_score_radial(params):
    pts = _radial_to_cartesian(params)
    s = _score(pts)
    if s <= 0:
        return 1e10
    return -s


def _neg_soft_score_radial(params, temperature):
    pts = _radial_to_cartesian(params)
    s = _soft_score(pts, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Seed generators ----------

def _seed_regular():
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


def _seed_quasi_regular(seed=0, ang_scale=0.05, rad_scale=0.05):
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang += rng.normal(0, ang_scale, N)
    rad = 1.0 + rng.normal(0, rad_scale, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.05, min_temp=1e-5, n_steps=10):
    """Anneal temperature, using L-BFGS-B + Powell at each level."""
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def _annealed_soft_minimize_radial(points, deadline, max_temp=0.05, min_temp=1e-5, n_steps=8):
    """Anneal temperature in radial/angular parameterization."""
    best = points.copy()
    best_s = _score(best)
    params = _cartesian_to_radial(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        if time.time() < deadline - 5:
            try:
                res = minimize(
                    lambda x: _neg_soft_score_radial(x, temp),
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                params = res.x
                cand = _radial_to_cartesian(params)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        if time.time() < deadline - 3:
            try:
                res = minimize(
                    lambda x: _neg_soft_score_radial(x, temp),
                    params,
                    method='Powell',
                    options={'maxiter': 60, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                params = res.x
                cand = _radial_to_cartesian(params)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using true objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    if time.time() < deadline - 3:
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
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search_random(points, deadline, n_iter=150, sigma_start=0.02, sigma_end=0.001):
    """Random-direction local search with adaptive step."""
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        for scale in [1.0, 0.5, 0.2, 2.0]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


def _coordinate_wise_local_search(points, deadline, step=0.01, max_passes=4):
    """Coordinate-wise line search."""
    best = points.copy()
    best_s = _score(best)
    
    for _pass in range(max_passes):
        if time.time() > deadline - 8:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                if time.time() > deadline - 8:
                    break
                for delta in [step, -step, step * 0.5, -step * 0.5, step * 2.0, -step * 2.0]:
                    cand = best.copy()
                    cand[i, dim] += delta
                    s = _score(cand)
                    if s > best_s + 1e-12:
                        best = cand
                        best_s = s
                        improved = True
                        break
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    
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

    # ---------- Generate diverse seeds ----------
    seeds = []
    seeds.append(_seed_regular())
    
    for r_in in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for off in [0.0, np.pi / 6, np.pi / 3, np.pi / 12]:
            seeds.append(_seed_two_ring(7, r_in, off))
    for r_in in [0.4, 0.5, 0.6, 0.7]:
        for off in [0.0, np.pi / 8, np.pi / 4]:
            seeds.append(_seed_two_ring(8, r_in, off))
    for r_in in [0.5, 0.6, 0.7]:
        for off in [0.0, np.pi / 5]:
            seeds.append(_seed_two_ring(6, r_in, off))
    
    for r2 in [0.5, 0.6, 0.7]:
        for r3 in [0.2, 0.3, 0.4]:
            for o2 in [0.0, np.pi / 6]:
                for o3 in [0.0, np.pi / 4]:
                    seeds.append(_seed_three_ring(6, 4, 3, r2, r3, o2, o3))
    for r2 in [0.5, 0.6]:
        for r3 in [0.25, 0.35]:
            seeds.append(_seed_three_ring(5, 5, 3, r2, r3, np.pi / 10, np.pi / 6))
    
    for s in range(6):
        seeds.append(_seed_quasi_regular(seed=s, ang_scale=0.05, rad_scale=0.05))
        seeds.append(_seed_quasi_regular(seed=s, ang_scale=0.1, rad_scale=0.1))

    # Score all seeds and keep top-6
    seed_scores = []
    for i, pts in enumerate(seeds):
        s = _score(pts)
        seed_scores.append((s, i, pts))
    seed_scores.sort(key=lambda x: -x[0])
    top_seeds = [pts for _, _, pts in seed_scores[:6]]

    for pts in top_seeds:
        _update(pts)

    # ---------- Phase 1: Annealed soft-min (temperature-based) from top seeds ----------
    for idx, seed_pts in enumerate(top_seeds[:4]):
        if time.time() > deadline - 40:
            break
        try:
            pts, s = _annealed_soft_minimize(seed_pts, deadline,
                                              max_temp=0.05, min_temp=1e-5, n_steps=8)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Penalty-based optimization from best ----------
    if best_points is not None and time.time() < deadline - 35:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=10.0, mu_end=1e5, n_mu_steps=6,
                                       max_iter_per_mu=80)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 3: Radial/angular parameterization optimization ----------
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _annealed_soft_minimize_radial(best_points, deadline,
                                                     max_temp=0.05, min_temp=1e-5, n_steps=7)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 4: Coordinate-wise local search ----------
    if best_points is not None and time.time() < deadline - 25:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.02, max_passes=4)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 5: Differential Evolution global search ----------
    if time.time() < deadline - 40:
        try:
            def de_obj(x):
                return _neg_score(x.reshape(N, 2))
            res = differential_evolution(
                de_obj,
                bounds=[(-1.5, 1.5)] * (2 * N),
                maxiter=50,
                popsize=18,
                tol=1e-7,
                seed=42,
                polish=True,
                init='sobol',
                disp=False,
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 6: Annealed soft-min from DE result ----------
    if best_points is not None and time.time() < deadline - 25:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline,
                                              max_temp=0.03, min_temp=1e-6, n_steps=7)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 7: Penalty-based from current best ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=50.0, mu_end=1e6, n_mu_steps=5,
                                       max_iter_per_mu=60)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 8: Random-direction local search ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _local_search_random(best_points, deadline, n_iter=100,
                                           sigma_start=0.015, sigma_end=0.0005)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 9: Radial optimization again ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _annealed_soft_minimize_radial(best_points, deadline,
                                                     max_temp=0.02, min_temp=1e-6, n_steps=6)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 10: Coordinate-wise fine search ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.005, max_passes=3)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 11: High-temperature anneal (escape) ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _annealed_soft_minimize(best_points, deadline,
                                              max_temp=0.1, min_temp=1e-5, n_steps=6)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 12: Final penalty optimization ----------
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=100.0, mu_end=1e7, n_mu_steps=4,
                                       max_iter_per_mu=50)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 13: True-objective polish (Powell + Nelder-Mead) ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 14: Final fine coordinate search ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.003, max_passes=2)
            _update(pts)
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
