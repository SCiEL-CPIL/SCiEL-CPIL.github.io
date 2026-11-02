"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)


def _cartesian_to_barycentric(pts):
    """Convert (x,y) to (b, c) barycentric coords where a=1-b-c."""
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    return np.column_stack([b, c])


def _barycentric_to_cartesian(bc):
    """Convert (b, c) barycentric coords to (x,y)."""
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * S3H
    return np.column_stack([x, y])


def _project_to_triangle(pts):
    """Vectorized projection into equilateral triangle."""
    y = np.clip(pts[:, 1], 0.0, S3H)
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(pts[:, 0], x_min, x_max)
    return np.column_stack([x, y])


def _min_triangle_area(pts):
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


def _hard_min_bary(flat_bc):
    """Objective in barycentric coords: -min_area."""
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    return -_min_triangle_area(pts)


def _soft_min_bary(flat_bc, beta):
    """Objective in barycentric coords: -soft_min_area."""
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    return -_soft_min_area(pts, beta)


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

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = v * S3H
        return np.array([x, y])

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    # Phase A: Differential Evolution (global)
    if _elapsed_ok(55):
        try:
            # DE in Cartesian with penalty
            def de_obj(flat):
                pts = flat.reshape(-1, 2)
                penalty = 0.0
                for p in pts:
                    x, y = p
                    if y < 0: penalty += abs(y) * 500
                    if y > S3H: penalty += (y - S3H) * 500
                    if x < 0: penalty += abs(x) * 500
                    if x > 1: penalty += (x - 1) * 500
                    lv = y - S3 * x
                    if lv > 0: penalty += lv * 500
                    rv = y + S3 * x - S3
                    if rv > 0: penalty += rv * 500
                return -_min_triangle_area(_project_to_triangle(pts)) + penalty

            bounds = [(0, 1), (0, S3H)] * N
            de_res = differential_evolution(
                de_obj, bounds, maxiter=40, popsize=15,
                seed=42, tol=1e-6, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Multi-start barycentric L-BFGS-B with annealed beta
    n_starts = 30
    bases = [init.copy()]
    for _ in range(15):
        bases.append(_start_from_base(init, 0.25))
    for _ in range(8):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())

    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0, 1200.0]

    for s in range(n_starts):
        if not _elapsed_ok(35):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.05)
        _update_best(pts)

        bc = _cartesian_to_barycentric(pts)
        flat_bc = bc.flatten()

        for beta in beta_schedule:
            if not _elapsed_ok(30):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _soft_min_bary(x, b),
                    flat_bc,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.clip(bc_opt[:, 0], 0, 1)
                c = np.clip(bc_opt[:, 1], 0, 1 - b)
                pts = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts)
                flat_bc = np.column_stack([b, c]).flatten()
            except Exception:
                pass

    # Phase C: Nelder-Mead polish in barycentric coords
    if best_pts is not None and _elapsed_ok(15):
        bc_best = _cartesian_to_barycentric(best_pts)
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(6):
                if not _elapsed_ok(8):
                    break
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                b = np.clip(pert_bc[:, 0], 0, 1)
                c = np.clip(pert_bc[:, 1], 0, 1 - b)
                flat = np.column_stack([b, c]).flatten()

                def nm_obj(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.clip(bc_x[:, 0], 0, 1)
                    cx = np.clip(bc_x[:, 1], 0, 1 - bx)
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_soft_min_area(px, 1000.0)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.clip(bc_res[:, 0], 0, 1)
                    c = np.clip(bc_res[:, 1], 0, 1 - b)
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    # Final fine polish
    if best_pts is not None and _elapsed_ok(5):
        bc_best = _cartesian_to_barycentric(best_pts)
        flat = bc_best.flatten()

        def final_nm(x):
            bc_x = x.reshape(-1, 2)
            bx = np.clip(bc_x[:, 0], 0, 1)
            cx = np.clip(bc_x[:, 1], 0, 1 - bx)
            px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
            return -_min_triangle_area(px)

        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
            bc_res = res.x.reshape(-1, 2)
            b = np.clip(bc_res[:, 0], 0, 1)
            c = np.clip(bc_res[:, 1], 0, 1 - b)
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
