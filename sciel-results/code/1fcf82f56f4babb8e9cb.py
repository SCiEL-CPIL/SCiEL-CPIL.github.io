"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math
import time


def _compute_covering_radius(points):
    """Compute the covering radius (in radians) as the max circumradius of convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            v = p2 - p1
            w = p3 - p1
            vv = np.dot(v, v)
            ww = np.dot(w, w)
            vw = np.dot(v, w)
            det = vv * ww - vw * vw
            if abs(det) < 1e-12:
                continue
            t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
            t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
            C_plane = p1 + t1 * v + t2 * w
            norm_C = np.linalg.norm(C_plane)
            if norm_C < 1e-12:
                continue
            C = C_plane / norm_C
            cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_r:
                max_r = R
        return max_r
    except Exception:
        return np.pi


def _simulated_annealing(points, time_limit=100.0):
    """Optimize using Simulated Annealing."""
    points = points.copy()
    current_energy = _compute_covering_radius(points)
    best_energy = current_energy
    best_points = points.copy()
    
    T = 0.1  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Pick a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a random move: add a small random vector and renormalize
        # The step size should be related to T
        step_size = 0.05 * T / 0.1  # Scale step with temperature
        if step_size < 1e-4:
            step_size = 1e-4
            
        random_vec = np.random.normal(0, 1, 3)
        random_vec /= np.linalg.norm(random_vec)
        
        new_point = original_point + step_size * random_vec
        new_point /= np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_energy = _compute_covering_radius(points)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = points.copy()
        else:
            # Revert move
            points[idx] = original_point
            
        T *= cooling_rate
        
    return best_points


def construct():
    # Start with Fibonacci spiral for a good initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Run Simulated Annealing
    points = _simulated_annealing(points, time_limit=110.0)
    
    # Final normalization
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points
# EVOLVE-BLOCK-END
