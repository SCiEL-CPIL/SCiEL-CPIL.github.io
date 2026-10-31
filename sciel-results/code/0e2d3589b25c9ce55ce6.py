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


# ---------- Support-line polygon parameterization ----------
_NORMALS = np.column_stack([
    np.cos(2 * np.pi * np.arange(N) / N),
    np.sin(2 * np.pi * np.arange(N) / N)
])


def _polygon_from_offsets(offsets, scale=1.0):
    offsets = np.asarray(offsets, dtype=np.float64)
    n = len(offsets)
    vertices = np.zeros((n, 2))
    for i in range(n):
        n1 = _NORMALS[(i - 1) % n]
        n2 = _NORMALS[i]
        d1 = offsets[(i - 1) % n]
        d2 = offsets[i]
        A = np.array([n1, n2])
        b = np.array([d1, d2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-12:
            vertices[i] = [0, 0]
        else:
            vertices[i, 0] = (d1 * A[1, 1] - A[0, 1] * d2) / det
            vertices[i, 1] = (A[0, 0] * d2 - d1 * A[1, 0]) / det
    return vertices * scale


def _is_valid_convex_polygon(verts):
    """Check if vertices form a valid convex polygon (all cross products same sign)."""
    n = len(verts)
    if n < 3:
        return False
    if np.any(np.abs(verts) > 1e6):
        return False
    cross_prods = np.zeros(n)
    for i in range(n):
        v1 = verts[(i + 1) % n] - verts[i]
        v2 = verts[(i + 2) % n] - verts[(i + 1) % n]
        cross_prods[i] = v1[0] * v2[1] - v1[1] * v2[0]
    # All must have the same sign (convex)
    if np.any(cross_prods == 0):
        return False
    return np.all(cross_prods > 0) or np.all(cross_prods < 0)


def _offsets_from_points(points):
    """Extract offsets from a convex polygon (points in CCW order)."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    centroid = np.mean(points, axis=0)
    pts = points - centroid
    offsets = np.zeros(n)
    for i in range(n):
        offsets[i] = np.max(pts @ _NORMALS[i])
    return offsets, centroid


def _score_from_offsets(offsets, scale=1.0):
    verts = _polygon_from_offsets(offsets, scale)
    if not _is_valid_convex_polygon(verts):
        return 0.0
    return _score(verts)


def _neg_score_offsets(offsets, scale=1.0):
    s = _score_from_offsets(offsets, scale)
    if s <= 0:
        return 1e10
    return -s


def _soft_score_offsets(offsets, scale=1.0, temperature=0.01):
    verts = _polygon_from_offsets(offsets, scale)
    if not _is_valid_convex_polygon(verts):
        return 0.0
    return _soft_score(verts, temperature)


def _neg_soft_score_offsets(offsets, scale=1.0, temperature=0.01):
    s = _soft_score_offsets(offsets, scale, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Epigraph with smooth constraints ----------

def _epigraph_objective(x, mu, temp_viol=1e-4):
    t = x[0]
    pts = x[1:].reshape(N, 2)
    areas = _triangle_areas_vectorized(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    violations = t - areas
    smooth_viol = temp_viol * np.log1p(np.exp(violations / temp_viol))
    penalty = np.sum(smooth_viol ** 2)
    return -t / hull_a + mu * penalty


def _penalty_optimize(points, deadline, mu_start=10.0, mu_end=1e6, n_mu_steps=8,
                      max_iter_per_mu=100):
    best = points.copy()
    best_s = _score(best)
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.95
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:] = best.flatten()
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    for mu in mus:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                lambda x: _epigraph_objective(x, mu),
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


def _seed_quasi_regular(seed=0, ang_scale=0.05, rad_scale=0.05):
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    ang += rng.normal(0, ang_scale, N)
    rad = 1.0 + rng.normal(0, rad_scale, N)
    return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])


def _seed_regular_offsets(scale=1.0):
    inradius = scale * np.cos(np.pi / N)
    return np.full(N, inradius)


def _seed_perturbed_offsets(seed=0, base_scale=1.0, noise=0.05):
    rng = np.random.RandomState(seed)
    base = np.full(N, base_scale * np.cos(np.pi / N))
    offsets = base + rng.normal(0, noise, N)
    return offsets


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.05, min_temp=1e-5, n_steps=10):
    best = points.copy()
    best_s = _score(best)
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                res = minimize(obj, best.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                res = minimize(obj2, best.flatten(), method='Powell',
                               options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-10})
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    return best, best_s


def _polish_true_objective(points, deadline):
    best = points.copy()
    best_s = _score(best)
    if time.time() < deadline - 6:
        try:
            res = minimize(lambda x: _neg_score(x.reshape(N, 2)), best.flatten(),
                           method='Powell', options={'maxiter': 300, 'xatol': 1e-11, 'fatol': 1e-11})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    if time.time() < deadline - 3:
        try:
            res = minimize(lambda x: _neg_score(x.reshape(N, 2)), best.flatten(),
                           method='Nelder-Mead',
                           options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    return best, best_s


def _local_search_random(points, deadline, n_iter=150, sigma_start=0.02, sigma_end=0.001):
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
                best_delta = 0.0
                for delta in [step, -step, step*0.5, -step*0.5, step*2.0, -step*2.0,
                              step*0.25, -step*0.25, step*4.0, -step*4.0]:
                    cand = best.copy()
                    cand[i, dim] += delta
                    s = _score(cand)
                    if s > best_s + 1e-12:
                        best_s = s
                        best = cand
                        best_delta = delta
                        improved = True
                if best_delta != 0.0:
                    lo = best_delta * 0.5
                    hi = best_delta * 1.5
                    for _ in range(12):
                        if time.time() > deadline - 8:
                            break
                        mid = (lo + hi) / 2.0
                        cand = best.copy()
                        cand[i, dim] += mid
                        s = _score(cand)
                        if s > best_s + 1e-14:
                            best_s = s
                            best = cand
                            lo = mid
                        else:
                            hi = mid
        if not improved:
            step *= 0.5
            if step < 1e-10:
                break
    return best, best_s


def _simulated_annealing(points, deadline, n_iter=300, t_start=0.05, t_end=0.001):
    best = points.copy()
    best_s = _score(best)
    current = points.copy()
    current_s = best_s
    rng = np.random.RandomState(777)
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        progress = it / max(n_iter - 1, 1)
        T = t_start * (t_end / t_start) ** progress
        n_move = rng.randint(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        perturb = rng.randn(n_move, 2) * 0.03
        cand = current.copy()
        for k, idx in enumerate(indices):
            cand[idx] += perturb[k]
        s = _score(cand)
        delta = s - current_s
        if delta > 0 or rng.rand() < np.exp(-delta / (T + 1e-15)):
            current = cand
            current_s = s
            if s > best_s:
                best_s = s
                best = cand.copy()
    return best, best_s


def _optimize_offsets(deadline, max_temp=0.02, min_temp=1e-5, n_steps=8, n_starts=5):
    """Primary optimization in offset space with multiple restarts."""
    best_verts = None
    best_s = 0.0
    rng = np.random.RandomState(42)
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))

    for start_idx in range(n_starts):
        if time.time() >= deadline - 10:
            break

        if start_idx == 0:
            offsets0 = _seed_regular_offsets()
        elif start_idx == 1:
            offsets0 = _seed_perturbed_offsets(seed=101, base_scale=1.0, noise=0.03)
        elif start_idx == 2:
            offsets0 = _seed_perturbed_offsets(seed=202, base_scale=1.0, noise=0.06)
        elif start_idx == 3:
            offsets0 = _seed_perturbed_offsets(seed=303, base_scale=1.0, noise=0.02)
        else:
            offsets0 = _seed_perturbed_offsets(seed=start_idx * 100 + 1,
                                                base_scale=1.0, noise=0.04)

        cur_offsets = offsets0.copy()
        cur_s = _score_from_offsets(cur_offsets)

        for temp in temps:
            if time.time() >= deadline - 8:
                break

            if time.time() < deadline - 5:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score_offsets(x, 1.0, temp),
                        cur_offsets,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12},
                    )
                    cur_offsets = res.x
                    verts = _polygon_from_offsets(cur_offsets)
                    s = _score(verts)
                    if s > cur_s:
                        cur_s = s
                    if s > best_s:
                        best_s = s
                        best_verts = verts.copy()
                except Exception:
                    pass

            if time.time() < deadline - 3:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score_offsets(x, 1.0, temp),
                        cur_offsets,
                        method='Powell',
                        options={'maxiter': 150, 'xatol': 1e-10, 'fatol': 1e-10},
                    )
                    cur_offsets = res.x
                    verts = _polygon_from_offsets(cur_offsets)
                    s = _score(verts)
                    if s > cur_s:
                        cur_s = s
                    if s > best_s:
                        best_s = s
                        best_verts = verts.copy()
                except Exception:
                    pass

        if time.time() < deadline - 4:
            try:
                res = minimize(
                    lambda x: _neg_score_offsets(x, 1.0),
                    cur_offsets,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True},
                )
                cur_offsets = res.x
                verts = _polygon_from_offsets(cur_offsets)
                s = _score(verts)
                if s > best_s:
                    best_s = s
                    best_verts = verts.copy()
            except Exception:
                pass

    if best_verts is None:
        best_verts = _polygon_from_offsets(_seed_regular_offsets())
        best_s = _score(best_verts)
    return best_verts, best_s


def _offsets_anneal_from_points(points, deadline, max_temp=0.05, min_temp=1e-6, n_steps=10):
    """Convert points to offsets, anneal in offset space, return best points."""
    offsets, centroid = _offsets_from_points(points)
    best_verts = points.copy()
    best_s = _score(points)
    cur_offsets = offsets.copy()
    cur_s = best_s

    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    for temp in temps:
        if time.time() >= deadline - 6:
            break
        if time.time() < deadline - 4:
            try:
                res = minimize(
                    lambda x: _neg_soft_score_offsets(x, 1.0, temp),
                    cur_offsets,
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cur_offsets = res.x
                verts = _polygon_from_offsets(cur_offsets)
                s = _score(verts)
                if s > cur_s:
                    cur_s = s
                if s > best_s:
                    best_s = s
                    best_verts = verts.copy()
            except Exception:
                pass
        if time.time() < deadline - 3:
            try:
                res = minimize(
                    lambda x: _neg_soft_score_offsets(x, 1.0, temp),
                    cur_offsets,
                    method='Powell',
                    options={'maxiter': 120, 'xatol': 1e-10, 'fatol': 1e-10},
                )
                cur_offsets = res.x
                verts = _polygon_from_offsets(cur_offsets)
                s = _score(verts)
                if s > cur_s:
                    cur_s = s
                if s > best_s:
                    best_s = s
                    best_verts = verts.copy()
            except Exception:
                pass
    return best_verts, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

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
    for r_in in [0.5, 0.6, 0.7]:
        for off in [0.0, np.pi / 8, np.pi / 4]:
            seeds.append(_seed_two_ring(8, r_in, off))
    for s in range(8):
        seeds.append(_seed_quasi_regular(seed=s, ang_scale=0.05, rad_scale=0.05))
        seeds.append(_seed_quasi_regular(seed=s, ang_scale=0.1, rad_scale=0.1))

    seed_scores = []
    for i, pts in enumerate(seeds):
        s = _score(pts)
        seed_scores.append((s, i, pts))
    seed_scores.sort(key=lambda x: -x[0])
    top_seeds = [pts for _, _, pts in seed_scores[:10]]

    for pts in top_seeds:
        _update(pts)

    # ---------- Phase 1: Offset-space optimization (PRIMARY) ----------
    if time.time() < deadline - 70:
        try:
            pts, s = _optimize_offsets(deadline, max_temp=0.04, min_temp=1e-5,
                                        n_steps=10, n_starts=5)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 2: Annealed soft-min from top seeds ----------
    for idx, seed_pts in enumerate(top_seeds[:4]):
        if time.time() > deadline - 50:
            break
        try:
            pts, s = _annealed_soft_minimize(seed_pts, deadline,
                                              max_temp=0.05, min_temp=1e-5, n_steps=8)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 3: Offset anneal from best ----------
    if best_points is not None and time.time() < deadline - 45:
        try:
            pts, s = _offsets_anneal_from_points(best_points, deadline,
                                                  max_temp=0.04, min_temp=1e-6, n_steps=10)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 4: Penalty-based from best ----------
    if best_points is not None and time.time() < deadline - 35:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=10.0, mu_end=1e5, n_mu_steps=6,
                                       max_iter_per_mu=80)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 5: Coordinate-wise ----------
    if best_points is not None and time.time() < deadline - 30:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.02, max_passes=5)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 6: DE global search ----------
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                return _neg_score(x.reshape(N, 2))
            res = differential_evolution(
                de_obj,
                bounds=[(-1.5, 1.5)] * (2 * N),
                maxiter=80,
                popsize=25,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                disp=False,
            )
            _update(res.x.reshape(N, 2))
        except Exception:
            pass

    # ---------- Phase 7: SA to escape ----------
    if best_points is not None and time.time() < deadline - 28:
        try:
            pts, s = _simulated_annealing(best_points, deadline, n_iter=200, t_start=0.04, t_end=0.002)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 8: Offset-space round 2 ----------
    if time.time() < deadline - 25:
        try:
            pts, s = _optimize_offsets(deadline, max_temp=0.02, min_temp=1e-6,
                                        n_steps=8, n_starts=4)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 9: Penalty-based again ----------
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=50.0, mu_end=1e6, n_mu_steps=5,
                                       max_iter_per_mu=60)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 10: Random-direction local search ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _local_search_random(best_points, deadline, n_iter=150,
                                           sigma_start=0.02, sigma_end=0.0003)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 11: Offset anneal from best ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _offsets_anneal_from_points(best_points, deadline,
                                                  max_temp=0.02, min_temp=1e-7, n_steps=8)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 12: Coordinate-wise fine ----------
    if best_points is not None and time.time() < deadline - 10:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.008, max_passes=4)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 13: Final penalty ----------
    if best_points is not None and time.time() < deadline - 8:
        try:
            pts, s = _penalty_optimize(best_points, deadline,
                                       mu_start=100.0, mu_end=1e7, n_mu_steps=4,
                                       max_iter_per_mu=50)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 14: True-objective polish ----------
    if best_points is not None and time.time() < deadline - 6:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            _update(pts)
        except Exception:
            pass

    # ---------- Phase 15: High-precision final polish ----------
    if best_points is not None and time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_points = cand.copy()
        except Exception:
            pass

    # ---------- Phase 16: Final fine coordinate search ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            pts, s = _coordinate_wise_local_search(best_points, deadline, step=0.003, max_passes=3)
            _update(pts)
        except Exception:
            pass

    if best_points is None:
        best_points = _seed_regular()

    return best_points
