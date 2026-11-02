"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

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
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
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
    Points must be inside the triangle.
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
    Soft minimum of triangle areas using log-sum-exp for numerical stability.
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


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
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
    
    # Initial configuration: High symmetry guess
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    init_uv = _cart_to_bary(init_cart)
    
    def _objective_bary(uv_flat, beta):
        """
        Objective function for L-BFGS-B in barycentric coordinates.
        Minimizes negative soft-min area.
        """
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
        """
        Run L-BFGS-B with annealing beta.
        """
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

    # Phase 1: Multi-start optimization in barycentric coordinates
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    # Generate starting configurations
    n_starts = 30
    for s in range(n_starts):
        if not _elapsed_ok(20):
            break
        
        if s == 0:
            uv_start = init_uv.copy()
        else:
            # Random barycentric points
            uv_start = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                uv_start[i] = [u, v]
        
        uv_opt = _optimize_bary(uv_start.flatten(), beta_schedule)
        cart_opt = _bary_to_cart(uv_opt)
        _update_best(cart_opt)
        
        # If we have a good result, try to polish it
        if best_pts is not None and _elapsed_ok(15):
            # Convert best to barycentric and do a final polish
            best_uv = _cart_to_bary(best_pts)
            best_uv = _optimize_bary(best_uv.flatten(), [200.0, 500.0, 1000.0])
            best_cart = _bary_to_cart(best_uv)
            _update_best(best_cart)

    # Phase 2: Local refinement around the best solution
    if best_pts is not None and _elapsed_ok(10):
        best_uv = _cart_to_bary(best_pts)
        # Try small perturbations in barycentric space
        for s in range(15):
            if not _elapsed_ok(5):
                break
            
            perturbed_uv = best_uv + rng.normal(0, 0.02, size=(N, 2))
            perturbed_uv = _project_to_simplex(perturbed_uv.flatten())
            
            perturbed_uv = _optimize_bary(perturbed_uv, [200.0, 500.0, 1000.0])
            perturbed_cart = _bary_to_cart(perturbed_uv)
            _update_best(perturbed_cart)

    if best_pts is None:
        return _bary_to_cart(init_uv)

    best_pts = _project_cartesian(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
