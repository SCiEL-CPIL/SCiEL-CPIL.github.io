"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 145  # safety margin from 150s limit

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
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    # For valid points, w >= 0. If w < 0, the point is outside, but we handle via penalty.
    # However, for area calculation, we just use the linear combination.
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))

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

def _objective_soft(uv_flat, beta, penalty_weight=1000.0):
    """Objective: minimize negative soft-min area + penalty for invalid barycentric coords."""
    pts = _bary_to_cart(uv_flat)
    
    # Penalty for u + v > 1 (w < 0)
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    viol_w = np.maximum(0, -w)
    # Also penalize u < 0 or v < 0? Bounds handle u,v in [0,1].
    # But if we use bounds [0,1] for u and v, w can be negative.
    penalty = penalty_weight * np.sum(viol_w ** 2)
    
    ma = _soft_min_area_cart(pts, beta)
    return -ma + penalty

def _objective_hard(uv_flat, penalty_weight=1000.0):
    """Objective: minimize negative hard-min area + penalty."""
    pts = _bary_to_cart(uv_flat)
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    viol_w = np.maximum(0, -w)
    penalty = penalty_weight * np.sum(viol_w ** 2)
    
    ma = _vectorized_min_area_cart(pts)
    return -ma + penalty

def _is_valid(uv_flat, tol=1e-6):
    """Check if all points are inside the triangle (w >= -tol)."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    if np.any(w < -tol):
        return False
    if np.any(u < -tol) or np.any(v < -tol):
        return False
    return True

def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        # Clip u, v to [0, 1]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u + v > 1:
            # Scale down to u+v=1
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_uv = None
    best_area = 0.0
    
    # Initial random configuration in barycentric coords
    uv0 = rng.random((N, 2))
    # Ensure valid start
    for i in range(N):
        if uv0[i, 0] + uv0[i, 1] > 1:
            scale = 1.0 / (uv0[i, 0] + uv0[i, 1])
            uv0[i] *= scale
    best_uv = uv0.flatten().copy()
    pts0 = _bary_to_cart(best_uv)
    best_area = _vectorized_min_area_cart(pts0)
    
    # Optimization Loop with multiple restarts
    # We will use a continuation method on beta for each restart.
    
    max_restarts = 50
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 20:
            break
            
        # Generate new random start
        if restart == 0:
            uv_start = best_uv.copy()
        else:
            uv_start = rng.random((N, 2)).flatten()
            # Project to valid simplex
            uv_start = _project_to_simplex(uv_start)
            
        # Continuation method
        current_uv = uv_start.copy()
        betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        
        for beta in betas:
            if time.time() - start_time > DEADLINE - 10:
                break
                
            try:
                # L-BFGS-B on barycentric coords
                # Bounds: u in [0, 1], v in [0, 1]
                bounds = [(0, 1), (0, 1)] * N
                
                result = minimize(
                    lambda x: _objective_soft(x, beta),
                    current_uv,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
                )
                current_uv = result.x
            except Exception:
                continue
        
        # Polish with Nelder-Mead on hard objective
        if time.time() - start_time > DEADLINE - 5:
            break
            
        try:
            # Project to ensure validity before NM
            current_uv = _project_to_simplex(current_uv)
            
            result = minimize(
                lambda x: _objective_hard(x),
                current_uv,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            opt_uv = result.x
        except Exception:
            opt_uv = current_uv
            
        # Project final result
        opt_uv = _project_to_simplex(opt_uv)
        
        if _is_valid(opt_uv):
            pts = _bary_to_cart(opt_uv)
            area = _vectorized_min_area_cart(pts)
            if area > best_area:
                best_area = area
                best_uv = opt_uv.copy()

    # Final check and return
    if best_uv is None:
        # Fallback
        uv_final = _project_to_simplex(rng.random((N, 2)).flatten())
        return _bary_to_cart(uv_final)
    
    # Ensure the final result is valid
    if not _is_valid(best_uv):
        best_uv = _project_to_simplex(best_uv)
        
    return _bary_to_cart(best_uv)

# EVOLVE-BLOCK-END
