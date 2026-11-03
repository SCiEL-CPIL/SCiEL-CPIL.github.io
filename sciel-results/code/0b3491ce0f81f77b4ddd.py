"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import ConvexHull

def _compute_covering_radius(points):
    try:
        hull = ConvexHull(points)
        max_rad = 0
        for simplex in hull.simplices:
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]
            n = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            cos_rad = np.dot(n, a)
            cos_rad = np.clip(cos_rad, -1.0, 1.0)
            rad = np.arccos(cos_rad)
            if rad > max_rad:
                max_rad = rad
        return np.degrees(max_rad)
    except Exception:
        return 180.0

def _objective(x, beta=20.0):
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1
    pts = pts / norms
    
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)
    
    iu = np.triu_indices(N, k=1)
    pair_angles = angles[iu]
    
    max_angle = np.max(pair_angles)
    exp_terms = np.exp(-beta * (pair_angles - max_angle))
    log_sum = np.log(np.sum(exp_terms))
    soft_min = -(1.0/beta) * log_sum - max_angle
    
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_score = _compute_covering_radius(x0)
    best_points = x0
    
    # Multi-start Nelder-Mead
    rng = np.random.default_rng(42)
    starts = [x0.flatten()]
    for _ in range(5):
        perturbation = rng.normal(0, 0.2, size=(N, 3))
        new_x = x0 + perturbation
        new_x = new_x / np.linalg.norm(new_x, axis=1, keepdims=True)
        starts.append(new_x.flatten())
        
    for x_start in starts:
        if time.time() > deadline:
            break
            
        res = minimize(_objective, x_start, method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if time.time() > deadline:
            break
            
        if res.success:
            pts = res.x.reshape(N, 3)
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
            score = _compute_covering_radius(pts)
            if score < best_score:
                best_score = score
                best_points = pts
                
    return best_points
# EVOLVE-BLOCK-END
