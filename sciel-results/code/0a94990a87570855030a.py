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

DEADLINE = 145

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


def _all_triangle_areas(pts):
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
    areas = _all_triangle_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _soft_min_grad(pts, beta):
    """Compute gradient of soft_min_area with respect to points."""
    areas = _all_triangle_areas(pts)
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    
    grad = np.zeros((N, 2))
    w = 0.5 * weights * sign_cross
    
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])
    
    for i in range(_N_TRIP):
        grad[_T_I[i]] += grad_pi[i]
        grad[_T_J[i]] += grad_pj[i]
        grad[_T_K[i]] += grad_pk[i]
    
    return grad


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


def _soft_min_bary_grad(flat_bc, beta):
    """Objective and gradient in barycentric coords: -soft_min_area."""
    bc = flat_bc.reshape(-1, 2)
    b = np.clip(bc[:, 0], 0, 1)
    c = np.clip(bc[:, 1], 0, 1 - b)
    pts = _barycentric_to_cartesian(np.column_stack([b, c]))
    val = -_soft_min_area(pts, beta)
    grad_pts = _soft_min_grad(pts, beta)
    
    # Chain rule: barycentric to cartesian
    # x = b + 0.5*c
    # y = c * S3H
    # d(x,y)/d(b,c) = [[1, 0.5], [0, S3H]]
    # grad_bc = J^T * grad_pts where J is the Jacobian
    # d(x)/db = 1, d(x)/dc = 0.5
    # d(y)/db = 0, d(y)/dc = S3H
    # So grad_b = grad_x, grad_c = 0.5*grad_x + S3H*grad_y
    grad_b = grad_pts[:, 0]
    grad_c = 0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1]
    grad_flat = np.column_stack([-grad_b, -grad_c]).flatten()
    return val, grad_flat


def _push_apart(pts, alpha=10.0):
    """Add a repulsion force that pushes points apart, weighted by inverse area."""
    areas = _all_triangle_areas(pts)
    min_a = np.min(areas)
    if min_a < 1e-12:
        min_a = 1e-12
    
    # Weight triangles that are close to the minimum more heavily
    weights = np.exp(-alpha * (areas - min_a) / min_a)
    weights = weights / np.sum(weights)
    
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0
    
    grad = np.zeros((N, 2))
    w = 0.5 * weights * sign_cross
    
    grad_pi = w[:, None] * np.column_stack([-dy2, dx2])
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])
    grad_pk = w[:, None] * np.column_stack([-dy1, dx1])
    
    for i in range(_N_TRIP):
        grad[_T_I[i]] += grad_pi[i]
        grad[_T_J[i]] += grad_pj[i]
        grad[_T_K[i]] += grad_pk[i]
    
    return grad


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

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

    # Structured initial configurations based on known good Heilbronn structures
    inits = []
    
    # Config 1: Corners + edge midpoints + interior
    init1 = np.array([
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
    inits.append(init1)
    
    # Config 2: More symmetric interior placement
    init2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H * 0.5],
        [0.25, S3H * 0.5],
        [0.75, S3H * 0.5],
        [0.5, S3H * 0.25],
        [0.5, S3H * 0.75],
        [0.5, S3H * 0.45],
    ], dtype=float)
    inits.append(init2)
    
    # Config 3: Based on 3x3 grid + corners
    init3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [1/3, 0.0],
        [2/3, 0.0],
        [0.5, S3H/3],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2],
        [0.333, S3H/3],
        [0.667, S3H/3],
    ], dtype=float)
    inits.append(init3)
    
    for init in inits:
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

    # Phase A: Multi-start with push-apart gradient + L-BFGS-B
    n_starts = 50
    bases = list(inits)
    for _ in range(25):
        base = rng.integers(0, len(inits))
        bases.append(_start_from_base(inits[base], 0.2))
    for _ in range(15):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())

    # Use a fixed high beta for the soft-min, combined with push-apart
    BETA_FIXED = 500.0

    def _objective_with_push(flat_bc, alpha=1.0):
        bc = flat_bc.reshape(-1, 2)
        b = np.clip(bc[:, 0], 0, 1)
        c = np.clip(bc[:, 1], 0, 1 - b)
        pts = _barycentric_to_cartesian(np.column_stack([b, c]))
        soft = _soft_min_area(pts, BETA_FIXED)
        push = np.sum(_push_apart(pts, alpha=10.0) ** 2) * 0.001
        return -(soft - push)

    def _objective_grad_with_push(flat_bc, alpha=1.0):
        bc = flat_bc.reshape(-1, 2)
        b = np.clip(bc[:, 0], 0, 1)
        c = np.clip(bc[:, 1], 0, 1 - b)
        pts = _barycentric_to_cartesian(np.column_stack([b, c]))
        
        soft = _soft_min_area(pts, BETA_FIXED)
        grad_soft = _soft_min_grad(pts, BETA_FIXED)
        push_grad = _push_apart(pts, alpha=10.0)
        
        val = -(soft - 0.001 * np.sum(push_grad ** 2))
        
        # Chain rule for barycentric
        grad_b = grad_soft[:, 0] - 0.002 * push_grad[:, 0]
        grad_c = 0.5 * grad_soft[:, 0] + S3H * grad_soft[:, 1] - 0.002 * push_grad[:, 1]
        
        return val, np.column_stack([-grad_b, -grad_c]).flatten()

    for s in range(n_starts):
        if not _elapsed_ok(40):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.03)
        _update_best(pts)

        bc = _cartesian_to_barycentric(pts)
        flat_bc = bc.flatten()

        # Run L-BFGS-B with combined objective
        try:
            res = minimize(
                lambda x: _objective_with_push(x)[0],
                flat_bc,
                jac=lambda x: _objective_grad_with_push(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30}
            )
            bc_opt = res.x.reshape(-1, 2)
            b = np.clip(bc_opt[:, 0], 0, 1)
            c = np.clip(bc_opt[:, 1], 0, 1 - b)
            pts = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(pts)
            flat_bc = np.column_stack([b, c]).flatten()
        except Exception:
            pass

        # Second pass with pure soft-min at higher beta
        try:
            res = minimize(
                lambda x: _soft_min_bary(x, 2000.0),
                flat_bc,
                jac=lambda x: _soft_min_bary_grad(x, 2000.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 20}
            )
            bc_opt = res.x.reshape(-1, 2)
            b = np.clip(bc_opt[:, 0], 0, 1)
            c = np.clip(bc_opt[:, 1], 0, 1 - b)
            pts = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(pts)
        except Exception:
            pass

    # Phase B: Nelder-Mead polish with exact min on best candidates
    if best_pts is not None and _elapsed_ok(20):
        bc_best = _cartesian_to_barycentric(best_pts)
        
        for scale in [0.03, 0.015, 0.008, 0.004]:
            if not _elapsed_ok(12):
                break
            for _ in range(10):
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
                    return -_min_triangle_area(px)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.clip(bc_res[:, 0], 0, 1)
                    c = np.clip(bc_res[:, 1], 0, 1 - b)
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                    # Update bc_best if improved
                    if _min_triangle_area(p) > _min_triangle_area(_barycentric_to_cartesian(bc_best)):
                        bc_best = np.column_stack([b, c]).copy()
                except Exception:
                    pass

    # Phase C: Final ultra-fine polish
    if best_pts is not None and _elapsed_ok(8):
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
                           options={'maxiter': 15000, 'xatol': 1e-14, 'fatol': 1e-16})
            bc_res = res.x.reshape(-1, 2)
            b = np.clip(bc_res[:, 0], 0, 1)
            c = np.clip(bc_res[:, 1], 0, 1 - b)
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    # Phase D: Coordinate descent on active set
    if best_pts is not None and _elapsed_ok(5):
        pts = best_pts.copy()
        for _ in range(50):
            if not _elapsed_ok(3):
                break
            areas = _all_triangle_areas(pts)
            min_a = np.min(areas)
            # Find points involved in minimum triangles
            active = set()
            for i, a in enumerate(areas):
                if a < min_a * 1.01:
                    active.add(_T_I[i])
                    active.add(_T_J[i])
                    active.add(_T_K[i])
            
            # Try small random perturbations of active points
            improved = False
            for pt_idx in active:
                for _ in range(3):
                    if not _elapsed_ok(2):
                        break
                    new_pts = pts.copy()
                    new_pts[pt_idx] += rng.normal(0, 0.005, size=2)
                    new_pts = _project_to_triangle(new_pts)
                    if _min_triangle_area(new_pts) > min_a:
                        pts = new_pts
                        improved = True
                        min_a = _min_triangle_area(pts)
                        break
            if not improved:
                break
        _update_best(pts)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return pts

    return _project_to_triangle(best_pts)
# EVOLVE-BLOCK-END
