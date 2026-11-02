"""Heilbronn triangles: maximize the smallest triangle area among 11 points.

Structural idea: represent the 11 points with a fixed high-symmetry base
configuration (concentric rings) and optimize a small set of symmetry-breaking
parameters (per-ring scale + rotation, plus a few radial jitter terms) instead
of all 22 coordinates independently.  This shrinks the search dimension from
22 to a handful, so the annealed-beta gradient machinery and a bounded local
polish can explore the space more effectively within the wall-clock budget.
"""
import time
import numpy as np
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 142  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])
CENTER = np.array([1.0 / 3.0, S3H / 3.0])

# Precompute all triplets for N=11 points (165 of them)
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])


def _min_triangle_area(pts):
    """Minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas via log-sum-exp (differentiable)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _project_to_triangle(pts):
    """Project points into the equilateral triangle (mutates in place)."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


# ---------------------------------------------------------------------------
# Symmetry-parameterized representation (the structural change).
#
#   p0  = centroid (fixed)
#   p1..p3   = outer ring (3 points),  scale s0, rotation a0
#   p4..p8   = middle ring (5 points), scale s1, rotation a1
#   p9,p10   = inner ring (2 points),  scale s2, rotation a2
#   e0,e1,e2 = small per-point radial jitter (breaks residual symmetry)
#
# => 3 scales + 3 rotations + 3 jitter = 9 parameters (vs 22 raw coords).
# ---------------------------------------------------------------------------
def _build_points(params):
    p = np.asarray(params, dtype=float)
    s0, s1, s2 = p[0], p[1], p[2]
    a0, a1, a2 = p[3], p[4], p[5]
    e0, e1, e2 = p[6], p[7], p[8]

    pts = np.empty((N, 2), dtype=float)

    # centroid
    pts[0] = CENTER

    # outer ring: 3 points, base angles 90,210,330 deg
    ang = np.deg2rad(np.array([90.0, 210.0, 330.0])) + a0
    rad = 0.55 * s0 * (1.0 + e0)
    pts[1, 0] = CENTER[0] + rad * np.cos(ang)
    pts[1, 1] = CENTER[1] + rad * np.sin(ang)
    pts[2, 0] = CENTER[0] + rad * np.cos(ang + 2 * np.pi / 3)
    pts[2, 1] = CENTER[1] + rad * np.sin(ang + 2 * np.pi / 3)
    pts[3, 0] = CENTER[0] + rad * np.cos(ang + 4 * np.pi / 3)
    pts[3, 1] = CENTER[1] + rad * np.sin(ang + 4 * np.pi / 3)

    # middle ring: 5 points, base angles 36,108,180,252,324 deg
    ang = np.deg2rad(np.array([36.0, 108.0, 180.0, 252.0, 324.0])) + a1
    rad = 0.28 * s1 * (1.0 + e1)
    pts[4, 0] = CENTER[0] + rad * np.cos(ang)
    pts[4, 1] = CENTER[1] + rad * np.sin(ang)
    pts[5, 0] = CENTER[0] + rad * np.cos(ang + 2 * np.pi / 5)
    pts[5, 1] = CENTER[1] + rad * np.sin(ang + 2 * np.pi / 5)
    pts[6, 0] = CENTER[0] + rad * np.cos(ang + 4 * np.pi / 5)
    pts[6, 1] = CENTER[1] + rad * np.sin(ang + 4 * np.pi / 5)
    pts[7, 0] = CENTER[0] + rad * np.cos(ang + 6 * np.pi / 5)
    pts[7, 1] = CENTER[1] + rad * np.sin(ang + 6 * np.pi / 5)
    pts[8, 0] = CENTER[0] + rad * np.cos(ang + 8 * np.pi / 5)
    pts[8, 1] = CENTER[1] + rad * np.sin(ang + 8 * np.pi / 5)

    # inner ring: 2 points, base angles 60,240 deg
    ang = np.deg2rad(np.array([60.0, 240.0])) + a2
    rad = 0.13 * s2 * (1.0 + e2)
    pts[9, 0] = CENTER[0] + rad * np.cos(ang)
    pts[9, 1] = CENTER[1] + rad * np.sin(ang)
    pts[10, 0] = CENTER[0] + rad * np.cos(ang + np.pi)
    pts[10, 1] = CENTER[1] + rad * np.sin(ang + np.pi)

    return pts


def _sym_soft_min(params, beta):
    pts = _build_points(params)
    _project_to_triangle(pts)
    return -_soft_min_area(pts, beta)


def _sym_hard_min(params):
    pts = _build_points(params)
    _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _farthest_point_sampling(rng, n, seed_pts):
    """Greedy farthest-point sampling to spread points inside the triangle."""
    pts = []
    # seed a few points first
    for _ in range(min(3, n)):
        u = rng.random(); v = rng.random()
        if u + v > 1:
            u = 1 - u; v = 1 - v
        a = 1 - u - v
        pts.append([a * 0 + u * 1 + v * 0.5, a * 0 + u * 0 + v * S3H])
    while len(pts) < n:
        cur = np.array(pts)
        d2 = np.min(np.sum((cur[:, None, :] - cur[None, :, :]) ** 2, axis=2), axis=1)
        # also consider distance to the three corners for spread
        corners = np.array([V0, V1, V2])
        dc2 = np.min(np.sum((cur[:, None, :] - corners[None, :, :]) ** 2, axis=2), axis=1)
        score = d2 + dc2
        best = int(np.argmax(score))
        u = rng.random(); v = rng.random()
        if u + v > 1:
            u = 1 - u; v = 1 - v
        a = 1 - u - v
        pts.append([a * 0 + u * 1 + v * 0.5, a * 0 + u * 0 + v * S3H])
        # nudge the farthest slot toward the chosen far point
        pts[best] = pts[-1]
    return np.array(pts)


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # ---- Initial symmetric configuration (all scales=1, rotations=0) ----
    init_params = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    init_pts = _project_to_triangle(_build_points(init_params).copy())
    _update_best(init_pts)

    # ---- Fallback spread configuration (farthest-point sampling) ----
    fsp = _farthest_point_sampling(rng, N, None)
    _update_best(fsp)

    # ---- Phase A: differential evolution over the 9 symmetry parameters ----
    # This is the global search in the reduced (9-D) space.
    if _elapsed_ok(70):
        try:
            bounds = [
                (0.5, 1.5), (0.5, 1.5), (0.5, 1.5),   # scales
                (-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6),  # rotations (rad)
                (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25),  # jitter
            ]
            de_res = differential_evolution(
                _sym_hard_min,
                bounds,
                maxiter=40,
                popsize=14,
                seed=7,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7,
            )
            de_pts = _project_to_triangle(_build_points(de_res.x).copy())
            _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: annealed-beta gradient optimization over symmetry params ----
    # Multi-start: from the best symmetric params, the DE result, and random
    # perturbations of the base.  Each start runs the beta-anneal schedule.
    n_starts = 16
    starts = [init_params.copy()]
    if best_pts is not None:
        # derive a rough parameterization seed by random scales/rotations
        for _ in range(8):
            sp = np.array([
                rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3),
                rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15),
            ])
            starts.append(sp)

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = starts[s % len(starts)]
        params = base + rng.normal(0, 0.05, size=9)
        pts = _project_to_triangle(_build_points(params).copy())
        _update_best(pts)

        for beta in [12.0, 35.0, 90.0, 220.0, 500.0, 1000.0]:
            if not _elapsed_ok(28):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _sym_soft_min(x, b),
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10},
                )
                params = res.x
                pts = _project_to_triangle(_build_points(params).copy())
                _update_best(pts)
            except Exception:
                pass

    # ---- Phase C: bounded full-coordinate polish on the best configuration ----
    # Now that we have a strong seed, refine all 22 coordinates locally (bounded
    # Nelder-Mead) to escape any residual symmetry-induced suboptimality.
    if best_pts is not None and _elapsed_ok(12):
        def _nm_obj(x):
            p = x.reshape(-1, 2)
            _project_to_triangle(p)
            return -_min_triangle_area(p)

        for scale in [0.04, 0.015, 0.006]:
            if not _elapsed_ok(9):
                break
            for _ in range(6):
                if not _elapsed_ok(8):
                    break
                cand = best_pts + rng.normal(0, scale, size=(N, 2))
                cand = _project_to_triangle(cand)
                try:
                    res = minimize(
                        _nm_obj,
                        cand.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11},
                    )
                    p = _project_to_triangle(res.x.reshape(-1, 2))
                    _update_best(p)
                except Exception:
                    pass

    # ---- Final fine polish on the absolute best ----
    if best_pts is not None and _elapsed_ok(5):
        def _nm_obj(x):
            p = x.reshape(-1, 2)
            _project_to_triangle(p)
            return -_min_triangle_area(p)
        try:
            res = minimize(
                _nm_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-13},
            )
            p = _project_to_triangle(res.x.reshape(-1, 2))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        # fallback random feasible configuration
        pts = np.array([_farthest_point_sampling(rng, N, None)[0]] +
                       [np.zeros(2) for _ in range(N - 1)])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
