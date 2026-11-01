import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _covering_radius(points):
    """
    Calculate the covering radius (max facet circumradius) for a set of points on the unit sphere.
    """
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-9):
        return np.pi
    points = points / norms[:, np.newaxis]
    
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-9:
                continue
            normal = normal / norm_n
            d = np.dot(normal, p1)
            if d <= 1e-9:
                return np.pi
            cos_R = np.clip(np.dot(normal, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def _polish(points, max_iters=500):
    """
    Gradient descent polish to refine the solution using repulsion forces.
    """
    alpha = 0.05
    for it in range(max_iters):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        mask = np.eye(N, dtype=bool)
        inv_dist_sq = np.zeros_like(dist_sq)
        np.divide(1.0, dist_sq, out=inv_dist_sq, where=~mask)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        dot_fp = np.sum(forces * points, axis=1)
        forces_tan = forces - dot_fp[:, np.newaxis] * points
        
        points = points + alpha * forces_tan
        
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, np.newaxis]
        
        alpha *= 0.99
        
    return points

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    current_energy = _covering_radius(points)
    best_points = points.copy()
    best_energy = current_energy
    
    # Simulated Annealing parameters
    T = 0.01
    T_min = 1e-6
    cooling_rate = 0.995
    max_iters = 50000
    start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > 100: # Reserve 20s for polish
            break
            
        idx = np.random.randint(N)
        p_old = points[idx].copy()
        
        rand_vec = np.random.normal(size=3)
        dot_prod = np.dot(rand_vec, p_old)
        tangent_vec = rand_vec - dot_prod * p_old
        norm_t = np.linalg.norm(tangent_vec)
        if norm_t < 1e-9:
            continue
        tangent_vec = tangent_vec / norm_t
        
        step_size = np.sqrt(T) * 0.5
        new_p = p_old + step_size * tangent_vec
        new_p = new_p / np.linalg.norm(new_p)
        
        points[idx] = new_p
        
        new_energy = _covering_radius(points)
        
        delta = new_energy - current_energy
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        else:
            points[idx] = p_old
            
        T *= cooling_rate
        
    # Polish the best found solution
    polished = _polish(best_points)
    polished_energy = _covering_radius(polished)
    
    if polished_energy < best_energy:
        best_points = polished
        best_energy = polished_energy
        
    return best_points
