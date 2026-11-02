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


def _soft_min_grad(pts, beta):
    """Compute gradient of soft_min_area with respect to points."""
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
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # d(areas)/d(pts) for each triplet
    # area = 0.5 * |cross|
    # d(area)/d(pi) = -0.5 * (dy2, -dx2) * sign(cross)
    # d(area)/d(pj) = 0.5 * (dy2, -dx2) * sign(cross) - 0.5 * (dy1, -dx1) * sign(cross)
    # d(area)/d(pk) = 0.5 * (dy1, -dx1) * sign(cross)
    
    # For |cross|, the gradient is sign(cross) * d(cross)/d(pts)
    # d(cross)/d(pi) = (-dy2, dx2)
    # d(cross)/d(pj) = (dy2, -dx2)
    # d(cross)/d(pk) = (-dy1, dx1)
    
    sign_cross = np.sign(cross)
    sign_cross[sign_cross == 0] = 1.0  # avoid zero
    
    grad = np.zeros((N, 2))
    w = 0.5 * weights * sign_cross
    
    # d(soft_min)/d(pts) = sum_k weights_k * d(area_k)/d(pts)
    # But soft_min = max_a - log_sum/beta, so d(soft_min)/d(pts) = -d(log_sum)/d(pts)/beta
    # d(log_sum)/d(pts) = (1/sum_exp) * sum_k exp_vals_k * d(areas_k)/d(pts)
    # So d(soft_min)/d(pts) = -sum_k weights_k * d(areas_k)/d(pts)
    
    # d(areas_k)/d(pi_k) = 0.5 * sign(cross_k) * (-dy2_k, dx2_k)
    # d(areas_k)/d(pj_k) = 0.5 * sign(cross_k) * (dy2_k, -dx2_k)
    # d(areas_k)/d(pk_k) = 0.5 * sign(cross_k) * (-dy1_k, dx1_k)
    
    # But we need the gradient of soft_min, which is -sum weights * d(area)/d(pts)
    # So:
    # d(soft_min)/d(pi) = -sum w_k * 0.5 * sign(cross_k) * (-dy2_k, dx2_k) = sum w_k * 0.5 * sign(cross_k) * (dy2_k, -dx2_k)
    # Wait, let me be more careful.
    
    # soft_min = max_a - (1/beta) * log(sum exp(-beta*(a_k - max_a)))
    # d(soft_min)/d(pts) = -(1/beta) * d(log(sum exp(...)))/d(pts)
    # = -(1/beta) * sum_k [exp(-beta*(a_k - max_a)) / sum_j exp(-beta*(a_j - max_a))] * d(-beta*(a_k - max_a))/d(pts)
    # = -sum_k weights_k * (-d(a_k)/d(pts))
    # = sum_k weights_k * d(a_k)/d(pts)
    
    # So d(soft_min)/d(pts) = sum_k weights_k * d(a_k)/d(pts)
    
    # d(a_k)/d(pi_k) = 0.5 * sign(cross_k) * (-dy2_k, dx2_k)
    # d(a_k)/d(pj_k) = 0.5 * sign(cross_k) * (dy2_k - dy1_k, -dx2_k + dx1_k)
    # Wait, that's not right either. Let me recompute.
    
    # a_k = 0.5 * |cross_k| where cross_k = dx1*dy2 - dy1*dx2
    # If cross > 0: a = 0.5 * cross
    # da/dpi = 0.5 * d(cross)/dpi = 0.5 * (-dy2, dx2)
    # da/dpj = 0.5 * d(cross)/dpj = 0.5 * (dy2, -dx2)
    # da/dpk = 0.5 * d(cross)/dpk = 0.5 * (-dy1, dx1)
    
    # If cross < 0: a = -0.5 * cross
    # da/dpi = -0.5 * (-dy2, dx2) = 0.5 * (dy2, -dx2)
    # da/dpj = -0.5 * (dy2, -dx2) = 0.5 * (-dy2, dx2)
    # da/dpk = -0.5 * (-dy1, dx1) = 0.5 * (dy1, -dx1)
    
    # So in general:
    # da/dpi = 0.5 * sign(cross) * (-dy2, dx2)
    # da/dpj = 0.5 * sign(cross) * (dy2, -dx2)
    # da/dpk = 0.5 * sign(cross) * (-dy1, dx1)
    
    # And d(soft_min)/d(pts) = sum_k weights_k * da_k/d(pts)
    
    # Accumulate contributions
    # For pi:
    grad_pi = w[:, None] * np.column_stack([dy2, -dx2])
    # For pj:
    grad_pj = w[:, None] * np.column_stack([dy2, -dx2])  # Wait, this should be (dy2, -dx2)
    # Actually: da/dpj = 0.5 * sign(cross) * (dy2, -dx2)
    # Hmm, let me double-check:
    # cross = dx1*dy2 - dy1*dx2
    # d(cross)/dpj_x = d(cross)/ddx1 = dy2
    # d(cross)/dpj_y = d(cross)/ddy1 = -dx2
    # So d(cross)/dpj = (dy2, -dx2)
    # da/dpj = 0.5 * sign(cross) * (dy2, -dx2)
    
    # For pk:
    # d(cross)/dpk_x = d(cross)/ddx2 = -dy1
    # d(cross)/dpk_y = d(cross)/ddy2 = dx1
    # So d(cross)/dpk = (-dy1, dx1)
    # da/dpk = 0.5 * sign(cross) * (-dy1, dx1)
    
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

    # Symmetric initial configuration - 11 points
    # Using a structured initial guess
    init = np.array([
        [0.0, 0.0],      # vertex 0
        [1.0, 0.0],      # vertex 1
        [0.5, S3H],      # vertex 2
        [0.5, 0.0],      # midpoint of bottom
        [0.75, S3H / 2], # midpoint of right edge
        [0.25, S3H / 2], # midpoint of left edge
        [1/3, S3H/3],    # interior
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
                de_obj, bounds, maxiter=50, popsize=20,
                seed=42, tol=1e-8, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Multi-start barycentric L-BFGS-B with annealed beta
    n_starts = 40
    bases = [init.copy()]
    for _ in range(20):
        bases.append(_start_from_base(init, 0.25))
    for _ in range(12):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())

    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0, 1200.0, 3000.0]

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
                # Use L-BFGS-B with gradient
                res = minimize(
                    lambda x, b=beta: _soft_min_bary(x, b),
                    flat_bc,
                    jac=lambda x, b=beta: _soft_min_bary_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 20}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.clip(bc_opt[:, 0], 0, 1)
                c = np.clip(bc_opt[:, 1], 0, 1 - b)
                pts = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts)
                flat_bc = np.column_stack([b, c]).flatten()
            except Exception:
                pass

    # Phase C: Nelder-Mead polish in barycentric coords with high beta
    if best_pts is not None and _elapsed_ok(15):
        bc_best = _cartesian_to_barycentric(best_pts)
        for scale in [0.04, 0.02, 0.01, 0.005]:
            if not _elapsed_ok(10):
                break
            for _ in range(8):
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
                    return -_soft_min_area(px, 2000.0)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 6000, 'xatol': 1e-11, 'fatol': 1e-13})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.clip(bc_res[:, 0], 0, 1)
                    c = np.clip(bc_res[:, 1], 0, 1 - b)
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    # Phase D: Final fine polish with exact min
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
                           options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15})
            bc_res = res.x.reshape(-1, 2)
            b = np.clip(bc_res[:, 0], 0, 1)
            c = np.clip(bc_res[:, 1], 0, 1 - b)
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    # Phase E: Additional local search with different perturbation patterns
    if best_pts is not None and _elapsed_ok(3):
        bc_best = _cartesian_to_barycentric(best_pts)
        # Try a few more perturbations
        for scale in [0.002, 0.001]:
            if not _elapsed_ok(2):
                break
            for _ in range(5):
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                b = np.clip(pert_bc[:, 0], 0, 1)
                c = np.clip(pert_bc[:, 1], 0, 1 - b)
                flat = np.column_stack([b, c]).flatten()

                def final2_nm(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.clip(bc_x[:, 0], 0, 1)
                    cx = np.clip(bc_x[:, 1], 0, 1 - bx)
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_min_triangle_area(px)

                try:
                    res = minimize(final2_nm, flat, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
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
