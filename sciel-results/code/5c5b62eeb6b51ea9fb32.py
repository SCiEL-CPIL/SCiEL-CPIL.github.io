"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

S3H = np.sqrt(3) / 2
TIME_LIMIT = 115  # seconds, safety margin

def _min_area_raw(points):
    """Compute the minimum triangle area among all triples of points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                if cross < min_a:
                    min_a = cross
    return min_a

def _soft_min_area(points, beta=100.0):
    """
    Smooth surrogate for the negative minimum triangle area using log-sum-exp.
    We minimize this, so we return -soft_min.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                dx2 = points[k, 0] - points[i, 0]
                dy2 = points[k, 1] - points[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2) * 0.5
                areas.append(cross)
    
    areas = np.array(areas)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we shift by the min
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    soft_min = min_a - np.log(log_sum) / beta
    return -soft_min

def _objective_bary_smooth(x, beta=100.0):
    """
    Objective for smooth optimization.
    x: flat array of 2*N values. Pairs (u, v) are barycentric coords.
    Returns: -soft_min_area + penalty.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Penalty for invalid points (u>=0, v>=0, u+v<=1)
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    penalty = 1000.0 * (np.sum(invalid_u**2) + np.sum(invalid_v**2) + np.sum(invalid_w**2))
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    return _soft_min_area(pts, beta) + penalty

def _objective_bary_hard(x):
    """
    Objective for hard min (used in final polish or DE).
    """
    u = x[0::2]
    v = x[1::2]
    
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    penalty = 1000.0 * (np.sum(invalid_u**2) + np.sum(invalid_v**2) + np.sum(invalid_w**2))
    
    pts = np.zeros((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    return -_min_area_raw(pts) + penalty

def _clamp_to_triangle(u, v):
    """Clamp barycentric coordinates to the valid triangle region."""
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        u[mask] /= sum_uv[mask]
        v[mask] /= sum_uv[mask]
    return u, v

def _bary_to_cart(u, v):
    """Convert barycentric coords to Cartesian."""
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def _is_valid(pts):
    """Check if points are inside the triangle."""
    px = pts[:, 0]
    py = pts[:, 1]
    w0 = 1.0 - px - py / S3H
    w1 = px - py / (2.0 * S3H)
    w2 = py / S3H
    return np.min(w0) >= -1e-6 and np.min(w1) >= -1e-6 and np.min(w2) >= -1e-6

def construct():
    start_time = time.time()
    
    best_points = None
    best_score = -float('inf')
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Stage 1: Differential Evolution for global search
    # Use hard objective for DE to avoid soft-min issues in global search
    try:
        res_de = differential_evolution(
            _objective_bary_hard, 
            bounds, 
            seed=42, 
            popsize=20, 
            maxiter=100, 
            tol=1e-8, 
            polish=False,
            init='sobol'
        )
        
        if time.time() > start_time + TIME_LIMIT - 20:
            if res_de.x is not None:
                u = res_de.x[0::2]
                v = res_de.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                if _is_valid(pts):
                    return pts
            return np.zeros((N, 2))
            
        if res_de.x is not None:
            best_x_de = res_de.x
        else:
            best_x_de = np.random.random(2*N)
            
    except Exception:
        best_x_de = np.random.random(2*N)

    # Stage 2: Local Polish with L-BFGS-B using smooth objective
    # We use the smooth objective to help L-BFGS-B navigate the kinks
    
    # Start with the best from DE
    candidates = [best_x_de]
    
    # Add a few random perturbations
    for _ in range(3):
        perturbed = best_x_de + np.random.normal(0, 0.02, 2*N)
        candidates.append(perturbed)
        
    # Add a few fully random starts
    for _ in range(2):
        candidates.append(np.random.random(2*N))

    for x0 in candidates:
        if time.time() > start_time + TIME_LIMIT - 5:
            break
            
        try:
            # Use L-BFGS-B with smooth objective
            res_lbfgs = minimize(
                lambda x: _objective_bary_smooth(x, beta=100.0), 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res_lbfgs.x is not None:
                u = res_lbfgs.x[0::2]
                v = res_lbfgs.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                
                if _is_valid(pts):
                    min_a = _min_area_raw(pts)
                    if min_a > best_score:
                        best_score = min_a
                        best_points = pts.copy()
                        
        except Exception:
            continue

    # Stage 3: Final polish with Nelder-Mead on hard objective
    if best_points is not None:
        # Convert best_points back to barycentric for polishing
        px = best_points[:, 0]
        py = best_points[:, 1]
        u = px - 0.5 * py / S3H
        v = py / S3H
        x_best = np.zeros(2*N)
        x_best[0::2] = u
        x_best[1::2] = v
        
        if time.time() < start_time + TIME_LIMIT - 2:
            try:
                res_nm = minimize(
                    _objective_bary_hard, 
                    x_best, 
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                if res_nm.x is not None:
                    u = res_nm.x[0::2]
                    v = res_nm.x[1::2]
                    u, v = _clamp_to_triangle(u, v)
                    pts = _bary_to_cart(u, v)
                    
                    if _is_valid(pts):
                        min_a = _min_area_raw(pts)
                        if min_a > best_score:
                            best_score = min_a
                            best_points = pts.copy()
                            
            except Exception:
                pass

    if best_points is None:
        # Fallback: Use DE result directly if it was valid
        if 'best_x_de' in locals():
            u = best_x_de[0::2]
            v = best_x_de[1::2]
            u, v = _clamp_to_triangle(u, v)
            best_points = _bary_to_cart(u, v)
        else:
            # Absolute fallback
            best_points = np.zeros((N, 2))
            for i in range(N):
                while True:
                    u, v = np.random.random(), np.random.random()
                    if u + v < 1.0:
                        break
                best_points[i, 0] = u + 0.5 * v
                best_points[i, 1] = S3H * v

    return best_points

# EVOLVE-BLOCK-END
