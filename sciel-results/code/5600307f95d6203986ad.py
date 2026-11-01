import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 50

def _compute_circumradii(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return [np.pi/2] * 100
    
    radii = []
    for face in hull.simplices:
        p1, p2, p3 = points[face]
        d12 = p1 - p2
        d13 = p1 - p3
        c_vec = np.cross(d12, d13)
        norm_c = np.linalg.norm(c_vec)
        if norm_c < 1e-10:
            radii.append(np.pi / 2)
            continue
        c_vec = c_vec / norm_c
        cos_r = np.dot(c_vec, p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        radii.append(np.arccos(cos_r))
    return radii

def _covering_radius(points):
    radii = _compute_circumradii(points)
    if not radii:
        return np.pi / 2
    return np.max(radii)

def _objective(points_flat, beta=100.0):
    points = points_flat.reshape(-1, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    radii = _compute_circumradii(points)
    if not radii:
        return np.pi / 2
    
    radii = np.array(radii)
    r_max = np.max(radii)
    if r_max < 1e-10:
        return 0.0
    val = np.log(np.sum(np.exp(beta * (radii - r_max)))) / beta + r_max
    return val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = np.pi / 2
    
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            points = init_points.copy()
        else:
            noise_scale = 0.2 * (0.8 ** restart)
            noise = np.random.normal(0, noise_scale, (N, 3))
            points = init_points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        points_flat = points.flatten()
        
        # Optimize
        res = minimize(_objective, points_flat, method='L-BFGS-B', 
                       options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
        
        points_new = res.x.reshape(-1, 3)
        points_new = points_new / np.linalg.norm(points_new, axis=1, keepdims=True)
        
        new_score = _covering_radius(points_new)
        if new_score < best_score:
            best_score = new_score
            best_points = points_new.copy()

    # Final polish
    if best_points is not None:
        points_flat = best_points.flatten()
        res = minimize(_objective, points_flat, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
        points_final = res.x.reshape(-1, 3)
        points_final = points_final / np.linalg.norm(points_final, axis=1, keepdims=True)
        
        final_score = _covering_radius(points_final)
        if final_score < best_score:
            best_points = points_final
            best_score = final_score

    return best_points
