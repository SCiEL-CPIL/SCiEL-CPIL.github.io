# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

N = 130

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def compute_covering_radius(points):
    """
    Computes the covering radius of the point configuration.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    simplices = hull.simplices
    m = len(simplices)
    max_radius = 0.0
    
    for simplex in simplices:
        a, b, c = points[simplex]
        v1 = a - b
        v2 = b - c
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-10:
            # Degenerate triangle, radius is effectively pi/2 or large?
            # If points are close, radius is small. If collinear, it's a great circle?
            # In convex hull, this shouldn't happen for generic points.
            continue
        
        o = normal / norm_n
        centroid = (a + b + c) / 3.0
        if np.dot(o, centroid) < 0:
            o = -o
            
        cos_r = np.clip(np.dot(o, a), -1.0, 1.0)
        radii = np.arccos(cos_r)
        if radii > max_radius:
            max_radius = radii
            
    return max_radius

def random_perturbation(points, idx, step_size):
    """
    Returns a new point for index idx, perturbed by step_size.
    """
    p = points[idx]
    # Generate random tangent vector
    # Pick random vector, project out p
    v = np.random.randn(3)
    v = v - np.dot(v, p) * p
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10:
        # Retry or pick orthogonal
        if abs(p[0]) < 0.9:
            v = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], p) * p
        else:
            v = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], p) * p
        v_norm = np.linalg.norm(v)
    
    v = v / v_norm * step_size
    new_p = p + v
    new_p = normalize(new_p.reshape(1,3))[0]
    return new_p

def construct():
    # Initial configuration
    points = fibonacci_sphere(N)
    points = normalize(points)
    
    current_radius = compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # Simulated Annealing Parameters
    T_start = 0.05 # Initial temperature (radians difference)
    T_end = 1e-6
    steps = 5000
    step_size = 0.1 # Base step size
    
    for i in range(steps):
        # Linear cooling
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Adapt step size: larger when T is high?
        # Or keep constant?
        # Let's try constant step size first, maybe scaled by T?
        # If T is high, we accept worse moves, so we can be bolder.
        current_step = step_size * (0.5 + T / T_start)
        
        # Pick random point
        idx = np.random.randint(N)
        
        # Save old point
        old_p = points[idx].copy()
        
        # Generate new point
        new_p = random_perturbation(points, idx, current_step)
        points[idx] = new_p
        
        # Evaluate new radius
        new_radius = compute_covering_radius(points)
        
        # Accept/Reject
        if new_radius < current_radius:
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Delta is new - old (positive)
            delta = new_radius - current_radius
            # Accept with probability exp(-delta / T)
            if np.random.rand() < np.exp(-delta / T):
                current_radius = new_radius
            else:
                # Reject
                points[idx] = old_p
                
    return best_points
# EVOLVE-BLOCK-END
