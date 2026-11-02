"""Heilbronn triangles: maximize the smallest triangle area among 11 points."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 138  # safety margin from 150s limit

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
_N_TRIP = len(_TRIPLETS)

# Barycentric coordinate conversion matrices
# Point = a*V0 + b*V1 + c*V2 where a+b+c=1, a,b,c >= 0
# x = b + c*0.5, y = c*S3H
# So: b = x - 0.5*c, c = y/S3H, a = 1 - b - c
def _cartesian_to_barycentric(pts):
    """Convert (x,y) points to (a,b,c) barycentric coords with a+b+c=1."""
    n = len(pts)
    abc = np.zeros((n, 3))
    c = pts[:, 1] / S3H
    b = pts[:, 0] - 0.5 * c
    a = 1.0 - b - c
    abc[:, 0] = a
    abc[:, 1] = b
    abc[:, 2] = c
    return abc

def _barycentric_to_cartesian(abc):
    """Convert (a,b,c) barycentric coords to (x,y) cartesian points."""
    n = len(abc)
    pts = np.zeros((n, 2))
    pts[:, 0] = abc[:, 1] + 0.5 * abc[:, 2]
    pts[:, 1] = abc[:, 2] * S3H
    return pts

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
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

def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return areas

def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _barycentric_penalty(abc):
    """Penalty for barycentric coords outside [0,1] or not summing to 1 (vectorized)."""
    neg = np.minimum(abc, 0.0)
    pos = np.maximum(abc - 1.0, 0.0)
    penalty = 100.0 * (np.sum(neg ** 2) + np.sum(pos ** 2))
    sums = abc.sum(axis=1)
    penalty += 100.0 * np.sum((sums - 1.0) ** 2)
    return penalty

def _objective_bary(flat, beta):
    """Objective in barycentric space: minimize -soft_min + penalty."""
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _soft_min_area(pts, beta)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _objective_bary_hard(flat):
    """Hard min objective in barycentric space."""
    abc = flat.reshape(-1, 3)
    pts = _barycentric_to_cartesian(abc)
    ma = _min_triangle_area(pts)
    pen = _barycentric_penalty(abc)
    return -ma + pen

def _push_apart(pts, alpha=1.0):
    """Gradient-free push: move each point away from others to increase min area (vectorized)."""
    n = len(pts)
    # Pairwise differences: (n, n, 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = pts[i] - pts[j]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero
    mask = dist > 1e-10
    dist_safe = np.where(mask, dist, 1.0)
    # Force magnitude: alpha / dist^2, direction: diff/dist
    force_mag = np.where(mask, alpha / (dist_safe ** 2), 0.0)
    # displacement[i] = sum_j force_mag[i,j] * diff[i,j] / dist_safe[i,j]
    disp = np.sum(force_mag[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis], axis=1)
    return pts + disp

def _project_to_triangle(pts):
    """Project points into the equilateral triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

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

    def _normalize_barycentric(abc):
        """Vectorized clip and normalize barycentric coords."""
        abc = np.clip(abc, 0.0, 1.0)
        sums = abc.sum(axis=1, keepdims=True)
        sums = np.where(sums > 1e-12, sums, 1.0)
        abc = abc / sums
        return abc

    # ---- Initial configuration: symmetric placement with better spread ----
    # Corners + edge midpoints + centroid + ring points
    init = np.array([
        [0.0, 0.0],           # corner 0
        [1.0, 0.0],           # corner 1
        [0.5, S3H],           # corner 2
        [0.5, 0.0],           # mid bottom edge
        [0.75, S3H / 2],      # mid right edge
        [0.25, S3H / 2],      # mid left edge
        [1/3, S3H/3],         # centroid
        [0.2, S3H/6],         # lower-left interior
        [0.8, S3H/6],         # lower-right interior
        [0.5, 2*S3H/3],       # upper interior
        [0.35, S3H/2.5],      # mid-left interior
    ], dtype=float)
    _update_best(init)

    # Additional diverse initial configs
    def _make_ring_config(scale=0.5):
        """Points on concentric rings in barycentric space."""
        pts = np.zeros((N, 2))
        # 3 corners
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        pts[2] = [0.5, S3H]
        # 3 edge midpoints
        pts[3] = [0.5, 0.0]
        pts[4] = [0.75, S3H / 2]
        pts[5] = [0.25, S3H / 2]
        # 5 interior points on two rings
        cx, cy = 1/3, S3H/3
        for i in range(3, 8):
            angle = 2 * np.pi * (i - 3) / 5
            r = scale * 0.3 if i < 6 else scale * 0.5
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle) * 0.8  # squish to fit triangle
            pts[i] = [x, y]
        return pts

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        a = 1 - u - v
        x = a * 0 + u * 1 + v * 0.5
        y = a * 0 + u * 0 + v * S3H
        return np.array([x, y])

    # Build diverse base configurations
    bases_cart = [init.copy()]
    bases_cart.append(_make_ring_config(0.6))
    bases_cart.append(_make_ring_config(0.4))
    for _ in range(8):
        bases_cart.append(init + rng.normal(0, 0.2, size=(N, 2)))
    for _ in range(4):
        bases_cart.append(np.array([_random_point_in_triangle() for _ in range(N)]))
    if best_pts is not None:
        bases_cart.append(best_pts.copy())

    # ---- Phase A: Differential Evolution in barycentric space ----
    if _elapsed_ok(45):
        try:
            def _de_obj_2d(flat_bc):
                """DE objective using 2D barycentric (b, c) pairs."""
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                # Vectorized penalty
                neg = np.minimum(abc, 0.0)
                pos = np.maximum(abc - 1.0, 0.0)
                pen = 500.0 * (np.sum(neg ** 2) + np.sum(pos ** 2))
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma + pen

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj_2d,
                bounds_de,
                maxiter=30,
                popsize=12,
                seed=42,
                tol=1e-5,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            bc = de_result.x.reshape(-1, 2)
            b = bc[:, 0]
            c = bc[:, 1]
            a = 1.0 - b - c
            abc = np.column_stack([a, b, c])
            abc = _normalize_barycentric(abc)
            de_pts = _barycentric_to_cartesian(abc)
            de_pts = _project_to_triangle(de_pts)
            _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Targeted soft-min gradient optimization ----
    # Use fewer starts but with adaptive beta scheduling
    n_starts = min(10, len(bases_cart))
    
    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases_cart[s].copy()
        pts = base + rng.normal(0, 0.03, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        abc = _cartesian_to_barycentric(pts)
        abc = _normalize_barycentric(abc)

        # Adaptive beta: start moderate, increase for sharpening
        betas = [50.0, 150.0, 500.0]
        for beta in betas:
            if not _elapsed_ok(35):
                break
            try:
                res = minimize(
                    lambda x, b=beta: _objective_bary(x, b),
                    abc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9}
                )
                abc_new = _normalize_barycentric(res.x.reshape(-1, 3))
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc = abc_new
            except Exception:
                pass

    # ---- Phase C: Iterative push-apart + Nelder-Mead local search ----
    if best_pts is not None and _elapsed_ok(25):
        for iteration in range(20):
            if not _elapsed_ok(12):
                break
            
            # Multiple push-apart steps with decreasing alpha
            cand = best_pts.copy()
            for alpha in [0.01, 0.005, 0.002]:
                cand = _push_apart(cand, alpha=alpha)
                cand = _project_to_triangle(cand)
            _update_best(cand)

            # Nelder-Mead polish from best
            try:
                abc0 = _cartesian_to_barycentric(best_pts)
                abc0 = _normalize_barycentric(abc0)
                res = minimize(
                    _objective_bary_hard,
                    abc0.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    # ---- Phase D: Final high-precision polish ----
    if best_pts is not None and _elapsed_ok(10):
        for _ in range(4):
            if not _elapsed_ok(5):
                break
            try:
                abc = _cartesian_to_barycentric(best_pts)
                abc = _normalize_barycentric(abc)
                res = minimize(
                    _objective_bary_hard,
                    abc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 8000, 'xatol': 1e-13, 'fatol': 1e-15}
                )
                abc_res = _normalize_barycentric(res.x.reshape(-1, 3))
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
            except Exception:
                pass

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
