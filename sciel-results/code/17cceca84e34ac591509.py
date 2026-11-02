"""Combined program for Heilbronn-11: Barycentric optimization with LP Polish."""
import numpy as np
import time
from scipy.optimize import minimize, linprog
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit. 
DEADLINE = 140.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
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


def _soft_min_area_cart(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp.
    """
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
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True

def _lp_polish_point(pts_cart, idx, current_min_area):
    """
    Polish a single point idx using Linear Programming.
    Fixes all other points.
    Maximizes the minimum area of triangles involving idx.
    """
    # Get fixed points
    fixed_indices = [i for i in range(N) if i != idx]
    P_fixed = pts_cart[fixed_indices]
    
    # Identify triangles involving idx
    # Triangles are (i, j, k). We need those where one of them is idx.
    # Since we precomputed _TRIPLETS, we can filter.
    # But it's faster to just iterate.
    # Let's find all triangles involving idx.
    # A triangle (a, b, c) involves idx if a==idx or b==idx or c==idx.
    
    # We will construct the LP for point idx.
    # Variables: u, v, t. (3 variables)
    # Constraints:
    # 1. u >= 0
    # 2. v >= 0
    # 3. u + v <= 1
    # 4. For each triangle (idx, a, b), Area(idx, a, b) >= t
    #    Area is linear in (u, v) with a sign.
    
    # Pre-calculate coefficients for linear area functions.
    # P_idx = (x, y) where x = 0.5 - 0.5u + 0.5v, y = S3H - S3H*u - S3H*v
    # Area = 0.5 * | det(P_a - P_idx, P_b - P_idx) |
    # Let P_idx = (x, y).
    # det = (x_a - x)(y_b - y) - (y_a - y)(x_b - x)
    #      = x_a y_b - x_a y - x y_b + xy - y_a x_b + y_a x + y x_b - xy
    #      = x_a y_b - y_a x_b + x (y_a - y_b) + y (x_b - x_a)
    # Let C0 = x_a y_b - y_a x_b
    # Let Cx = y_a - y_b
    # Let Cy = x_b - x_a
    # det = C0 + x * Cx + y * Cy
    # Substitute x, y in terms of u, v:
    # x = 0.5 - 0.5u + 0.5v
    # y = S3H - S3H*u - S3H*v
    # det = C0 + (0.5 - 0.5u + 0.5v)*Cx + (S3H - S3H*u - S3H*v)*Cy
    #     = C0 + 0.5*Cx + S3H*Cy 
    #       + u*(-0.5*Cx - S3H*Cy)
    #       + v*(0.5*Cx - S3H*Cy)
    
    # Let A_u = -0.5*Cx - S3H*Cy
    # Let A_v = 0.5*Cx - S3H*Cy
    # Let A_0 = C0 + 0.5*Cx + S3H*Cy
    # det = A_0 + A_u * u + A_v * v
    
    # Area = 0.5 * | det |
    # Constraint: 0.5 * | A_0 + A_u*u + A_v*v | >= t
    # Let L(u,v) = A_0 + A_u*u + A_v*v
    # We assume the sign of L(u,v) is consistent with the current configuration.
    # Current L_val = A_0 + A_u*u_curr + A_v*v_curr
    # sign = 1 if L_val >= 0 else -1
    # Constraint: sign * L(u,v) >= 2*t  (since Area >= t => 0.5|det| >= t => |det| >= 2t)
    # sign * (A_0 + A_u*u + A_v*v) >= 2*t
    # sign*A_u*u + sign*A_v*v - 2*t >= -sign*A_0
    
    # LP Form: Minimize c^T x subject to A_ub x <= b_ub, A_eq x = b_eq
    # We want to Maximize t. So Minimize -t.
    # Variables x = [u, v, t]
    # c = [0, 0, -1]
    
    # Inequalities A_ub x <= b_ub:
    # 1. -u <= 0
    # 2. -v <= 0
    # 3. u + v <= 1
    # 4. For each triangle: -sign*A_u*u - sign*A_v*v + 2*t <= sign*A_0
    
    # Let's collect constraints.
    A_ub = []
    b_ub = []
    
    # Box constraints
    A_ub.append([-1, 0, 0]); b_ub.append(0)
    A_ub.append([0, -1, 0]); b_ub.append(0)
    A_ub.append([1, 1, 0]); b_ub.append(1)
    
    # Area constraints
    # We need the current u, v for idx to determine signs.
    # pts_cart is Cartesian. We need u, v.
    # Let's convert current point to bary.
    x_curr, y_curr = pts_cart[idx]
    w_curr = y_curr / S3H
    v_curr = x_curr - 0.5 * w_curr
    u_curr = 1.0 - v_curr - w_curr
    
    # Iterate through all pairs (a, b) from fixed indices
    for i_a in range(len(fixed_indices)):
        for i_b in range(i_a + 1, len(fixed_indices)):
            a = fixed_indices[i_a]
            b = fixed_indices[i_b]
            x_a, y_a = pts_cart[a]
            x_b, y_b = pts_cart[b]
            
            C0 = x_a * y_b - y_a * x_b
            Cx = y_a - y_b
            Cy = x_b - x_a
            
            A_u = -0.5 * Cx - S3H * Cy
            A_v = 0.5 * Cx - S3H * Cy
            A_0 = C0 + 0.5 * Cx + S3H * Cy
            
            # Calculate current determinant value to get sign
            L_curr = A_0 + A_u * u_curr + A_v * v_curr
            
            # If L_curr is very close to 0, the point is on the edge. 
            # This is a degenerate case. We can pick an arbitrary sign or skip.
            # For robustness, if abs(L_curr) < 1e-9, we might have a problem.
            # Let's use a small epsilon bias.
            if abs(L_curr) < 1e-9:
                sign = 1.0
            else:
                sign = 1.0 if L_curr > 0 else -1.0
            
            # Constraint: sign * (A_0 + A_u*u + A_v*v) >= 2*t
            # => -sign*A_u*u - sign*A_v*v + 2*t <= sign*A_0
            
            A_ub.append([-sign * A_u, -sign * A_v, 2.0])
            b_ub.append(sign * A_0)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, -1.0])
    
    # Solve LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='HiGHS')
        if res.success:
            u_new, v_new, t_new = res.x
            # Check if t_new is an improvement
            # The global min area is min(t_new, min_area_of_other_triangles)
            # But we only optimized for triangles involving idx.
            # We must ensure we don't violate the global min area.
            # Actually, if t_new > current_min_area, it's a potential improvement for this point.
            # But we need to check if the new position is valid and if the sign assumption holds.
            
            # Verify sign assumption
            L_new = A_0 + A_u * u_new + A_v * v_new # This is for the last triangle, need to check all?
            # It's better to just calculate the new areas and check if min area increased.
            
            # Construct new points
            new_pts = pts_cart.copy()
            w_new = 1.0 - u_new - v_new
            x_new = v_new + 0.5 * w_new
            y_new = w_new * S3H
            new_pts[idx] = [x_new, y_new]
            
            # Project to be safe
            new_pts = _project_cartesian(new_pts)
            
            # Calculate new global min area
            new_min_area = _vectorized_min_area_cart(new_pts)
            
            # Accept if improved
            if new_min_area > current_min_area + 1e-10:
                return new_pts, new_min_area
    except Exception:
        pass
        
    return pts_cart, current_min_area

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Initial configuration
    init_cart = np.array([
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
    
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    init_uv = _cart_to_bary(init_cart)
    
    def _objective_bary(uv_flat, beta):
        pts = _bary_to_cart(uv_flat)
        uv = uv_flat.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        viol = np.maximum(0, -w)
        penalty = 1000.0 * np.sum(viol ** 2)
        ma = _soft_min_area_cart(pts, beta)
        return -ma + penalty

    def _optimize_bary(uv_start, beta_schedule):
        uv_curr = uv_start.copy()
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
            try:
                bounds = [(0, 1), (0, 1)] * N
                result = minimize(
                    lambda x: _objective_bary(x, beta),
                    uv_curr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                uv_curr = result.x
                uv_curr = _project_to_simplex(uv_curr)
            except Exception:
                break
        return uv_curr

    def _project_to_simplex(uv_flat):
        uv = uv_flat.reshape(-1, 2).copy()
        for i in range(len(uv)):
            u, v = uv[i]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            uv[i] = [u, v]
        return uv.flatten()

    # Phase 1: Multi-start L-BFGS-B
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    starts = [init_uv.flatten()]
    for _ in range(4):
        uv_rand = rng.random((N, 2))
        uv_rand = _project_to_simplex(uv_rand.flatten())
        starts.append(uv_rand)
        
    for start_uv in starts:
        if not _elapsed_ok(20):
            break
        opt_uv = _optimize_bary(start_uv, beta_schedule)
        pts = _bary_to_cart(opt_uv)
        pts = _project_cartesian(pts)
        _update_best(pts)

    # Phase 2: LP Polish
    if best_pts is not None and _elapsed_ok(50):
        current_pts = best_pts.copy()
        current_area = best_area
        improvement = True
        while improvement and _elapsed_ok(10):
            improvement = False
            for i in range(N):
                if not _elapsed_ok(10):
                    break
                new_pts, new_area = _lp_polish_point(current_pts, i, current_area)
                if new_area > current_area:
                    current_pts = new_pts
                    current_area = new_area
                    _update_best(current_pts)
                    improvement = True

    if best_pts is None:
        return init_cart
    return best_pts

# EVOLVE-BLOCK-END
