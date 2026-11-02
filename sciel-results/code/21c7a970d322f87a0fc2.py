"""Combined program for Heilbronn-11: LP-based max-min area with Nelder-Mead polish."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog, minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _bary_to_cart(uv_flat):
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2)."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v."""
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
    """Soft minimum of triangle areas."""
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
    """Project points into the triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u + v > 1:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


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


def _build_lp_matrix():
    """
    Build the constraint matrix for the LP formulation.
    Variables: x_0, y_0, x_1, y_1, ..., x_10, y_10, t (total 2N+1 = 23 vars)
    
    Constraints:
    1. For each triangle (i,j,k): area >= t
       area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
       We need to handle the absolute value. Since we're maximizing t,
       we can use the fact that at the optimum, the orientation is fixed.
       Actually, for LP we need linear constraints. The area formula:
       area = 0.5 * |(x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)|
       
       This is not linear due to the absolute value and the product terms.
       Wait - the area of a triangle with vertices at points i, j, k is:
       0.5 * |det([P_j - P_i, P_k - P_i])|
       = 0.5 * |(x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)|
       
       This is bilinear in the coordinates, not linear! So a direct LP won't work.
       
       Let me reconsider. The problem is that the area is a quadratic (bilinear) 
       function of the point coordinates. So this is NOT a linear program.
       
       However, we can reformulate. If we fix the orientation (sign of the cross product),
       then the constraint becomes linear. But we don't know the orientation a priori.
       
       Alternative: Use the fact that for a set of points, we can enumerate possible 
       orderings or use a different approach.
       
       Actually, let me think about this differently. The standard Heilbronn problem 
       formulation with LPs typically works when the points are constrained to a grid 
       or when using a different parameterization.
       
       For now, let me use a different structural approach: 
       - Use the soft-min annealing approach but with more aggressive multi-start
       - Add a local search phase that specifically optimizes the worst triangles
    """
    pass


def _optimize_worst_triangles(pts, n_iter=50):
    """
    Local optimization that focuses on the worst triangles.
    For each iteration, find the triangle with minimum area and try to increase it.
    """
    pts = pts.copy()
    
    for _ in range(n_iter):
        # Find all triangle areas
        pi = pts[_T_I]
        pj = pts[_T_J]
        pk = pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        # Find the worst triangle
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Get the three points of the worst triangle
        i, j, k = _TRIPLETS[min_idx]
        
        # Try to increase the area by moving points
        # Compute gradients with respect to each of the three points
        # area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
        # Let's assume positive orientation for gradient computation
        pi_p = pts[i]
        pj_p = pts[j]
        pk_p = pts[k]
        
        # Gradient of cross product w.r.t. P_i, P_j, P_k
        # cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # d(cross)/d(x_i) = -(y_k - y_i) + (y_j - y_i) = y_j - y_k
        # d(cross)/d(y_i) = (x_k - x_i) - (x_j - x_i) = x_k - x_j
        # d(cross)/d(x_j) = (y_k - y_i)
        # d(cross)/d(y_j) = -(x_k - x_i)
        # d(cross)/d(x_k) = -(y_j - y_i)
        # d(cross)/d(y_k) = (x_j - x_i)
        
        sign = np.sign(cross[min_idx]) if cross[min_idx] != 0 else 1.0
        
        grad_i = np.array([sign * (pts[j][1] - pts[k][1]), 
                           sign * (pts[k][0] - pts[j][0])])
        grad_j = np.array([sign * (pts[k][1] - pts[i][1]), 
                           sign * (-(pts[k][0] - pts[i][0]))])
        grad_k = np.array([sign * (-(pts[j][1] - pts[i][1])), 
                           sign * (pts[j][0] - pts[i][0])])
        
        # Normalize gradients
        norm_i = np.linalg.norm(grad_i)
        norm_j = np.linalg.norm(grad_j)
        norm_k = np.linalg.norm(grad_k)
        
        if norm_i > 1e-10:
            grad_i /= norm_i
        if norm_j > 1e-10:
            grad_j /= norm_j
        if norm_k > 1e-10:
            grad_k /= norm_k
        
        # Move points in the direction that increases the area
        step = 0.005
        pts[i] += step * grad_i
        pts[j] += step * grad_j
        pts[k] += step * grad_k
        
        # Project back to triangle
        pts = _project_cartesian(pts)
    
    return pts


def _nelder_mead_polish(pts, max_iter=2000, tol=1e-10):
    """Nelder-Mead optimization to polish the configuration."""
    def objective(x):
        p = x.reshape(-1, 2)
        p = _project_cartesian(p)
        return -_vectorized_min_area_cart(p)
    
    try:
        result = minimize(
            objective,
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol * 0.1, 'adaptive': True}
        )
        p = result.x.reshape(-1, 2)
        p = _project_cartesian(p)
        return p
    except Exception:
        return pts


def _lbfgs_bary(uv_start, beta_schedule, max_iter_per_beta=100):
    """L-BFGS-B optimization in barycentric coordinates with annealing."""
    uv_curr = uv_start.copy()
    
    for beta in beta_schedule:
        def objective(x):
            pts = _bary_to_cart(x)
            uv = x.reshape(-1, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            viol_w = np.maximum(0, -w)
            penalty = 500.0 * np.sum(viol_w ** 2)
            ma = _soft_min_area_cart(pts, beta)
            return -ma + penalty
        
        bounds = [(0, 1), (0, 1)] * N
        
        try:
            result = minimize(
                objective,
                uv_curr,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            uv_curr = result.x
            uv_curr = _project_to_simplex(uv_curr)
        except Exception:
            break
    
    return uv_curr


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
    
    # Initial configuration with good symmetry
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
    init = _project_cartesian(init)
    _update_best(init)
    
    # Phase 1: Multi-start with L-BFGS-B in barycentric coords
    n_starts = 25
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        
        # Generate perturbed starting point
        if s == 0:
            pts = init.copy()
        else:
            scale = 0.15 + 0.1 * (s % 5)
            pts = init + rng.normal(0, scale, size=(N, 2))
            pts = _project_cartesian(pts)
        
        _update_best(pts)
        
        # Optimize in barycentric coordinates
        uv = _cart_to_bary(pts)
        uv = _lbfgs_bary(uv, beta_schedule, max_iter_per_beta=80)
        pts = _bary_to_cart(uv)
        pts = _project_cartesian(pts)
        _update_best(pts)
    
    # Phase 2: Worst-triangle focused local search
    if best_pts is not None and _elapsed_ok(20):
        for _ in range(10):
            if not _elapsed_ok(15):
                break
            cand = best_pts + rng.normal(0, 0.03, size=(N, 2))
            cand = _project_cartesian(cand)
            cand = _optimize_worst_triangles(cand, n_iter=30)
            _update_best(cand)
    
    # Phase 3: Nelder-Mead polish on best configuration
    if best_pts is not None and _elapsed_ok(10):
        # Multiple NM runs with different initial perturbations
        for scale in [0.02, 0.01, 0.005]:
            if not _elapsed_ok(5):
                break
            cand = best_pts + rng.normal(0, scale, size=(N, 2))
            cand = _project_cartesian(cand)
            cand = _nelder_mead_polish(cand, max_iter=3000, tol=1e-11)
            _update_best(cand)
    
    # Phase 4: Final fine polish
    if best_pts is not None and _elapsed_ok(3):
        cand = _nelder_mead_polish(best_pts, max_iter=5000, tol=1e-12)
        _update_best(cand)
    
    # Final validation and return
    if best_pts is None:
        # Fallback
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = v + 0.5 * (1 - u - v)
            y = (1 - u - v) * S3H
            pts[i] = [x, y]
        return pts
    
    best_pts = _project_cartesian(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
