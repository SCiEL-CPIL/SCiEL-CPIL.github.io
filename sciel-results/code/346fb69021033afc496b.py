"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import random
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

def _objective_bary(x):
    """
    Objective function for Differential Evolution and Nelder-Mead.
    x: flat array of 2*N values. Pairs (u, v) are barycentric coords.
    Returns: -min_area + penalty.
    """
    u = x[0::2]
    v = x[1::2]
    
    # Check validity: u>=0, v>=0, u+v<=1
    # Penalty for invalid points
    invalid_u = np.maximum(0, -u)
    invalid_v = np.maximum(0, -v)
    invalid_w = np.maximum(0, u + v - 1.0)
    penalty = 100.0 * (np.sum(invalid_u) + np.sum(invalid_v) + np.sum(invalid_w))
    
    # Convert to Cartesian
    pts = np.zeros((N, 2))
    pts[:, 0] = u + 0.5 * v
    pts[:, 1] = S3H * v
    
    min_a = _min_area_raw(pts)
    return -min_a + penalty

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

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_points = None
    best_score = -float('inf')
    
    bounds = [(0.0, 1.0) for _ in range(2 * N)]
    
    # Stage 1: Differential Evolution for global search
    # We disable polish here to control the local search ourselves
    try:
        res = differential_evolution(
            _objective_bary, 
            bounds, 
            seed=42, 
            popsize=25, 
            maxiter=150, 
            tol=1e-8, 
            polish=False,
            init='sobol'
        )
        
        if time.time() > start_time + TIME_LIMIT - 5:
            # Out of time, return best from DE if valid
            if res.x is not None:
                u = res.x[0::2]
                v = res.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                min_a = _min_area_raw(pts)
                return pts
            return np.zeros((N, 2))
            
        if res.x is not None:
            best_x_de = res.x
        else:
            best_x_de = np.random.random(2*N)
            
    except Exception as e:
        best_x_de = np.random.random(2*N)

    # Stage 2: Local Polish with Nelder-Mead
    # We take the best solution from DE and polish it.
    # We also try a few random perturbations of the best solution.
    
    candidates = [best_x_de]
    
    # Add some random perturbations to explore local neighborhood
    for _ in range(5):
        perturbed = best_x_de + np.random.normal(0, 0.05, 2*N)
        candidates.append(perturbed)
        
    # Add a few fully random starts in case DE failed
    for _ in range(3):
        candidates.append(np.random.random(2*N))

    for x0 in candidates:
        if time.time() > start_time + TIME_LIMIT - 5:
            break
            
        try:
            res_nm = minimize(
                _objective_bary, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if res_nm.x is not None:
                u = res_nm.x[0::2]
                v = res_nm.x[1::2]
                u, v = _clamp_to_triangle(u, v)
                pts = _bary_to_cart(u, v)
                
                # Verify points are inside
                px = pts[:, 0]
                py = pts[:, 1]
                w0 = 1.0 - px - py / S3H
                w1 = px - py / (2.0 * S3H)
                w2 = py / S3H
                if np.min(w0) < -1e-6 or np.min(w1) < -1e-6 or np.min(w2) < -1e-6:
                    continue
                
                min_a = _min_area_raw(pts)
                if min_a > best_score:
                    best_score = min_a
                    best_points = pts.copy()
                    
        except Exception as e:
            continue

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
                    u, v = rng.random(), rng.random()
                    if u + v < 1.0:
                        break
                best_points[i, 0] = u + 0.5 * v
                best_points[i, 1] = S3H * v

    return best_points

# EVOLVE-BLOCK-END
