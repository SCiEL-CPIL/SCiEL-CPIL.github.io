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

DEADLINE = 140  # safety margin from 150s limit

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
    # Clip to avoid underflow/overflow issues, though exp(-large) is 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

def _epigraph_objective(x, t, lambda_val):
    """
    Epigraph objective: -t + lambda * sum(max(0, t - area_i)^2)
    x: flattened barycentric coords (3*N)
    t: scalar
    lambda_val: penalty weight
    """
    abc = x[:3*N].reshape(-1, 3)
    # Normalize barycentric coords to be valid
    abc = np.clip(abc, 0, 1)
    sums = abc.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    abc = abc / sums
    
    pts = _barycentric_to_cartesian(abc)
    areas = _all_triangle_areas(pts)
    
    # Penalty for constraint violation
    violations = np.maximum(0, t - areas)
    penalty = lambda_val * np.sum(violations**2)
    
    # Return -t + penalty
    return -t + penalty

def _barycentric_penalty(abc):
    """Penalty for barycentric coords outside [0,1] or not summing to 1."""
    penalty = 0.0
    # Each coordinate should be in [0,1]
    # We handle normalization in the objective, but this is for the hard min NM
    # For soft min, we rely on the projection/normalization in the objective or post-processing
    return 0.0 # We will handle constraints via projection/normalization

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

    # ---- Initial configuration: symmetric placement ----
    init = np.array([
        [0.0, 0.0],          # corner 0
        [1.0, 0.0],          # corner 1
        [0.5, S3H],          # corner 2
        [0.5, 0.0],          # mid bottom
        [0.75, S3H / 2],     # mid right
        [0.25, S3H / 2],     # mid left
        [1/3, S3H/3],        # centroid
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    # ---- Phase A: Differential Evolution in barycentric space (Quick) ----
    if _elapsed_ok(40):
        try:
            def _de_obj_2d(flat_bc):
                bc = flat_bc.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                # Clip to valid region
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                pts = _barycentric_to_cartesian(abc)
                ma = _min_triangle_area(pts)
                return -ma

            bounds_de = [(0, 1), (0, 1)] * N
            de_result = differential_evolution(
                _de_obj_2d,
                bounds_de,
                maxiter=20, # Reduced maxiter to save time
                popsize=10,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            if de_result.fun < 0:
                bc = de_result.x.reshape(-1, 2)
                b = bc[:, 0]
                c = bc[:, 1]
                a = 1.0 - b - c
                abc = np.column_stack([a, b, c])
                abc = np.clip(abc, 0, 1)
                sums = abc.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc = abc / sums
                de_pts = _barycentric_to_cartesian(abc)
                de_pts = _project_to_triangle(de_pts)
                _update_best(de_pts)
        except Exception:
            pass

    # ---- Phase B: Barycentric soft-min gradient optimization (Annealing) ----
    n_starts = 10
    bases_cart = [init.copy()]
    for _ in range(5):
        bases_cart.append(init + rng.normal(0, 0.25, size=(N, 2)))
    if best_pts is not None:
        bases_cart.append(best_pts.copy())

    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases_cart[s % len(bases_cart)]
        pts = base + rng.normal(0, 0.05, size=(N, 2))
        pts = _project_to_triangle(pts)
        _update_best(pts)

        # Convert to barycentric for optimization
        abc = _cartesian_to_barycentric(pts)
        # Clip to valid barycentric
        for i in range(N):
            for j in range(3):
                abc[i, j] = max(0.0, min(1.0, abc[i, j]))
            s_val = abc[i].sum()
            if s_val > 0:
                abc[i] /= s_val

        # Annealing schedule for beta
        for beta in [10.0, 30.0, 100.0, 300.0, 1000.0]:
            if not _elapsed_ok(20):
                break
            try:
                def _obj_soft(x, b=beta):
                    abc_local = x.reshape(-1, 3)
                    abc_local = np.clip(abc_local, 0, 1)
                    sums = abc_local.sum(axis=1, keepdims=True)
                    sums[sums == 0] = 1
                    abc_local = abc_local / sums
                    pts_local = _barycentric_to_cartesian(abc_local)
                    return -_soft_min_area(pts_local, b)

                res = minimize(
                    _obj_soft,
                    abc.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                abc_new = res.x.reshape(-1, 3)
                abc_new = np.clip(abc_new, 0, 1)
                sums = abc_new.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_new = abc_new / sums
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                _update_best(pts_new)
                abc = abc_new
            except Exception:
                pass

    # ---- Phase C: Epigraph-based Refinement (Lagrangian-like) ----
    if best_pts is not None and _elapsed_ok(30):
        abc_best = _cartesian_to_barycentric(best_pts)
        abc_best = np.clip(abc_best, 0, 1)
        sums = abc_best.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_best = abc_best / sums
        
        t_init = _min_triangle_area(best_pts)
        
        # Optimize (x, t) with increasing lambda
        lambda_schedule = [1.0, 10.0, 100.0, 1000.0]
        
        for lam in lambda_schedule:
            if not _elapsed_ok(15):
                break
            try:
                x0 = np.concatenate([abc_best.flatten(), [t_init]])
                
                def _obj_epigraph(x_full, l=lam):
                    return _epigraph_objective(x_full, x_full[-1], l)
                
                res = minimize(
                    _obj_epigraph,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                x_new = res.x
                abc_new = x_new[:3*N].reshape(-1, 3)
                t_new = x_new[-1]
                
                abc_new = np.clip(abc_new, 0, 1)
                sums = abc_new.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_new = abc_new / sums
                pts_new = _barycentric_to_cartesian(abc_new)
                pts_new = _project_to_triangle(pts_new)
                
                # Update t to be the actual min area for next step
                t_init = _min_triangle_area(pts_new)
                abc_best = abc_new
                _update_best(pts_new)
            except Exception:
                pass

    # ---- Phase D: Nelder-Mead Swarm (Aggressive Local Search) ----
    if best_pts is not None and _elapsed_ok(20):
        abc_best = _cartesian_to_barycentric(best_pts)
        abc_best = np.clip(abc_best, 0, 1)
        sums = abc_best.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        abc_best = abc_best / sums

        def _obj_hard_nm(x):
            abc_local = x.reshape(-1, 3)
            abc_local = np.clip(abc_local, 0, 1)
            sums = abc_local.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1
            abc_local = abc_local / sums
            pts_local = _barycentric_to_cartesian(abc_local)
            return -_min_triangle_area(pts_local)

        # Run multiple Nelder-Mead restarts
        nm_restarts = 30
        for i in range(nm_restarts):
            if not _elapsed_ok(5):
                break
            
            # Perturb the best barycentric coords
            noise = rng.normal(0, 0.01, size=(N, 3))
            x0 = abc_best.flatten() + noise.flatten()
            
            try:
                res = minimize(
                    _obj_hard_nm,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                abc_res = res.x.reshape(-1, 3)
                abc_res = np.clip(abc_res, 0, 1)
                sums = abc_res.sum(axis=1, keepdims=True)
                sums[sums == 0] = 1
                abc_res = abc_res / sums
                p = _barycentric_to_cartesian(abc_res)
                p = _project_to_triangle(p)
                _update_best(p)
                
                # Update abc_best if improved
                if _min_triangle_area(p) > best_area - 1e-9:
                    abc_best = abc_res
            except Exception:
                pass

    if best_pts is None:
        pts = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
            [0.5, 0.0], [0.75, S3H/2], [0.25, S3H/2],
            [1/3, S3H/3], [0.333, 0.0], [0.667, 0.0],
            [0.5, S3H/3], [0.2, 0.1]
        ])
        return pts

    best_pts = _project_to_triangle(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
