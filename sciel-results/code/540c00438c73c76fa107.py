"""
Combined program: ring-family global search (3-ring 5-4-5 + 2-ring 7-7) via DE,
then aggressive coordinate-level polish with soft-min continuation + Powell.
The key structural change is using multi-ring parameterizations for the global
search (fewer parameters, better-conditioned landscape) and then escaping the
ring constraint via full 28-DOF coordinate polish.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14

# Precompute triangle indices
_i_idx, _j_idx, _k_idx = [], [], []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _i_idx.append(i)
            _j_idx.append(j)
            _k_idx.append(k)
_I = np.array(_i_idx)
_J = np.array(_j_idx)
_K = np.array(_k_idx)

# Base angles for regular 14-gon
_BASE_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS_A = np.cos(_BASE_ANGLES)
_SIN_A = np.sin(_BASE_ANGLES)


def _all_triangle_areas(pts):
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 1e-10


def _score(pts):
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 0.0
    return np.min(_all_triangle_areas(pts)) / ha


def _soft_min(areas, beta):
    m = np.min(areas)
    shifted = areas - m
    max_shift = np.max(shifted)
    if max_shift > 50.0:
        return m
    return m - (1.0 / beta) * np.log(np.sum(np.exp(-beta * shifted)))


def _make_ring(n_pts, radius, angle_offset=0.0):
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + angle_offset
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _three_ring_545(r1, r2, a1, a2, a3):
    pts1 = _make_ring(5, r1, a1)
    pts2 = _make_ring(4, r2, a2)
    pts3 = _make_ring(5, 1.0, a3)
    return np.vstack([pts1, pts2, pts3])


def _three_ring_455(r1, r2, a1, a2, a3):
    pts1 = _make_ring(4, r1, a1)
    pts2 = _make_ring(5, r2, a2)
    pts3 = _make_ring(5, 1.0, a3)
    return np.vstack([pts1, pts2, pts3])


def _two_ring_77(r, alpha):
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = angles_outer + alpha
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    return np.vstack([outer_pts, inner_pts])


def _neg_soft_full(x, beta):
    pts = x.reshape(-1, 2)
    ha = _hull_area(pts)
    if ha < 1e-9:
        return 1e10
    areas = _all_triangle_areas(pts)
    return -(_soft_min(areas, beta) / ha)


def _neg_score_full(x):
    pts = x.reshape(-1, 2)
    return -_score(pts)


def _polish(pts, deadline):
    """Multi-stage local polish: soft-min continuation L-BFGS-B then Powell."""
    best_pts = pts.copy()
    best_s = _score(best_pts)
    x = best_pts.flatten()

    betas = [10.0, 40.0, 150.0, 500.0]
    for beta in betas:
        if time.time() > deadline - 6:
            break
        try:
            res = minimize(_neg_soft_full, x, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
                x = best_pts.flatten()
        except Exception:
            pass

    if time.time() < deadline - 3:
        try:
            res = minimize(_neg_score_full, x, method='Powell',
                           options={'maxiter': 400, 'xtol': 1e-10, 'ftol': 1e-12})
            s = -res.fun
            if s > best_s:
                best_s = s
                best_pts = res.x.reshape(-1, 2)
        except Exception:
            pass

    return best_pts, best_s


def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    best_pts = None
    best_score = -np.inf

    def update(pts):
        nonlocal best_pts, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        return s

    # 1. Regular 14-gon baseline
    reg_pts = np.column_stack([_COS_A, _SIN_A])
    update(reg_pts)

    # 2. Three-ring 5-4-5 DE
    if time.time() < deadline - 55:
        def _neg_545(params):
            r1, r2, a1, a2, a3 = params
            return -_score(_three_ring_545(r1, r2, a1, a2, a3))

        try:
            bounds = [(0.1, 0.65), (0.45, 0.9), (0.0, 2 * np.pi / 5),
                      (0.0, 2 * np.pi / 4), (0.0, 2 * np.pi / 5)]
            res = differential_evolution(_neg_545, bounds, seed=42,
                                         maxiter=120, tol=1e-8, polish=True,
                                         popsize=15)
            if res.fun < 0:
                update(_three_ring_545(*res.x))
        except Exception:
            pass

    # 3. Three-ring 4-5-5 DE
    if time.time() < deadline - 45:
        def _neg_455(params):
            r1, r2, a1, a2, a3 = params
            return -_score(_three_ring_455(r1, r2, a1, a2, a3))

        try:
            bounds = [(0.1, 0.55), (0.5, 0.88), (0.0, 2 * np.pi / 4),
                      (0.0, 2 * np.pi / 5), (0.0, 2 * np.pi / 5)]
            res = differential_evolution(_neg_455, bounds, seed=123,
                                         maxiter=100, tol=1e-8, polish=True,
                                         popsize=14)
            if res.fun < 0:
                update(_three_ring_455(*res.x))
        except Exception:
            pass

    # 4. Two-ring 7-7 DE
    if time.time() < deadline - 35:
        def _neg_77(params):
            r, alpha = params
            return -_score(_two_ring_77(r, alpha))

        try:
            bounds = [(0.2, 0.9), (0.0, 2 * np.pi / 7)]
            res = differential_evolution(_neg_77, bounds, seed=77,
                                         maxiter=100, tol=1e-8, polish=True,
                                         popsize=12)
            if res.fun < 0:
                update(_two_ring_77(*res.x))
        except Exception:
            pass

    # 5. Polish the best found (full 28-DOF coordinate polish)
    if best_pts is not None and time.time() < deadline - 20:
        best_pts, best_score = _polish(best_pts, deadline - 10)
        update(best_pts)

    # 6. Try polishing each ring candidate separately if time allows
    candidates = []
    # Re-evaluate ring configs quickly
    if time.time() < deadline - 30:
        # Re-run quick DE for 5-4-5 with different seed to get a second candidate
        def _neg_545b(params):
            r1, r2, a1, a2, a3 = params
            return -_score(_three_ring_545(r1, r2, a1, a2, a3))
        try:
            bounds = [(0.15, 0.6), (0.5, 0.88), (0.0, 2 * np.pi / 5),
                      (0.0, 2 * np.pi / 4), (0.0, 2 * np.pi / 5)]
            res = differential_evolution(_neg_545b, bounds, seed=99,
                                         maxiter=80, tol=1e-7, polish=True,
                                         popsize=12)
            if res.fun < 0:
                candidates.append(_three_ring_545(*res.x))
        except Exception:
            pass

    for cand in candidates:
        if time.time() > deadline - 15:
            break
        update(cand)
        if time.time() < deadline - 10:
            polished, _ = _polish(cand, deadline - 8)
            update(polished)

    # 7. Random restarts from perturbed regular polygon
    rng = np.random.RandomState(42)
    for restart in range(4):
        if time.time() > deadline - 18:
            break
        noise = rng.normal(0, 0.12, size=(N, 2))
        init = reg_pts + noise
        update(init)
        if time.time() < deadline - 12:
            polished, _ = _polish(init, deadline - 8)
            update(polished)

    # 8. Final polish
    if best_pts is not None and time.time() < deadline - 5:
        best_pts, best_score = _polish(best_pts, deadline)
        update(best_pts)

    if best_pts is None:
        return reg_pts
    return best_pts
