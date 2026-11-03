"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # Safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_covering_radius(pts):
    """
    Compute the covering radius of the configuration.
    Returns the max angular distance from any point on the sphere to its nearest neighbor.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
        
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-10:
            continue
            
        n = n / norm_n
        
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
            
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        radius = np.arccos(dot_val)
        
        if radius > max_radius:
            max_radius = radius
            
    return max_radius

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses Simulated Annealing with periodic reheating.
    """
    # Initialize with Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    current_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    current_pts = _to_sphere(current_pts)
    
    best_pts = current_pts.copy()
    best_radius = _compute_covering_radius(best_pts)
    current_radius = best_radius
    
    # SA Parameters
    T_current = 0.5
    T_min = 1e-5
    cooling_rate = 0.998
    
    rng = np.random.default_rng(42)
    last_improvement_time = _START_TIME
    
    while time.time() < _DEADLINE:
        # Reheating logic
        if time.time() - last_improvement_time > 10.0 and T_current < 0.1:
            T_current = 0.2  # Reheat to explore
        
        idx = rng.integers(0, N)
        
        # Perturbation
        noise_scale = T_current * 0.5
        noise = rng.normal(0, noise_scale, 3)
        new_pt = current_pts[idx] + noise
        norm = np.linalg.norm(new_pt)
        if norm < 1e-6:
            continue
        new_pt = new_pt / norm
        
        new_pts = current_pts.copy()
        new_pts[idx] = new_pt
        
        new_radius = _compute_covering_radius(new_pts)
        
        delta = new_radius - current_radius
        
        # Metropolis Acceptance
        if delta < 0 or (T_current > 0 and rng.random() < np.exp(-delta / T_current)):
            current_pts = new_pts
            current_radius = new_radius
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
                last_improvement_time = time.time()
        
        # Cooling
        T_current *= cooling_rate
        if T_current < T_min:
            T_current = T_min
            
    return _to_sphere(best_pts)

# EVOLVE-BLOCK-END
