"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _angular_distance(p, q):
    # p, q are unit vectors
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)

def _circumcenter_sphere(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle.
    Returns the unit vector of the circumcenter and the radius (angular distance to vertices).
    """
    # Side lengths (angular)
    a = _angular_distance(p2, p3)
    b = _angular_distance(p1, p3)
    c_ = _angular_distance(p1, p2)
    
    cos_a = np.cos(a)
    cos_b = np.cos(b)
    cos_c = np.cos(c_)
    sin_b = np.sin(b)
    sin_c = np.sin(c_)
    
    # Handle degenerate cases
    if sin_b * sin_c < 1e-10:
        # Degenerate triangle, return centroid
        centroid = _normalize(p1 + p2 + p3)
        return centroid, 0.0
        
    cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
    cos_A = np.clip(cos_A, -1.0, 1.0)
    A = np.arccos(cos_A)
    
    cos_B = (cos_b - cos_a * cos_c) / (np.sin(a) * sin_c)
    cos_B = np.clip(cos_B, -1.0, 1.0)
    B = np.arccos(cos_B)
    
    cos_C = (cos_c - cos_a * cos_b) / (np.sin(a) * sin_b)
    cos_C = np.clip(cos_C, -1.0, 1.0)
    C = np.arccos(cos_C)
    
    # Circumcenter vector
    c_vec = np.sin(A) * p1 + np.sin(B) * p2 + np.sin(C) * p3
    c_norm = np.linalg.norm(c_vec)
    if c_norm < 1e-10:
        centroid = _normalize(p1 + p2 + p3)
        return centroid, 0.0
    c_unit = c_vec / c_norm
    
    # Radius is the angular distance from circumcenter to any vertex
    radius = _angular_distance(c_unit, p1)
    
    return c_unit, radius

def _is_point_in_spherical_triangle(p, p1, p2, p3):
    """
    Check if point p is inside the spherical triangle p1, p2, p3.
    """
    def _side(p_edge1, p_edge2, p_test, p_ref):
        n = np.cross(p_edge1, p_edge2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            return True # Degenerate edge
        n = n / n_norm
        s_test = np.dot(p_test, n)
        s_ref = np.dot(p_ref, n)
        return (s_test * s_ref) >= 0

    if not _side(p1, p2, p, p3): return False
    if not _side(p2, p3, p, p1): return False
    if not _side(p3, p1, p, p2): return False
    return True

def _covering_radius(points):
    """
    Compute the covering radius and the worst facet (indices).
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: Use KDTree for rough estimate if hull fails
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points)
        return np.pi, 0, 1, 2

    max_radius = 0.0
    worst_i, worst_j, worst_k = 0, 1, 2
    worst_cc = None
    
    for simplex in hull.simplices:
        i, j, k = simplex
        p1, p2, p3 = points[i], points[j], points[k]
        cc, r = _circumcenter_sphere(p1, p2, p3)
        
        if _is_point_in_spherical_triangle(cc, p1, p2, p3):
            current_r = r
        else:
            d12 = _angular_distance(p1, p2)
            d23 = _angular_distance(p2, p3)
            d31 = _angular_distance(p3, p1)
            
            if d12 >= d23 and d12 >= d31:
                current_r = d12 / 2.0
            elif d23 >= d12 and d23 >= d31:
                current_r = d23 / 2.0
            else:
                current_r = d31 / 2.0
                
        if current_r > max_radius:
            max_radius = current_r
            worst_i, worst_j, worst_k = i, j, k
            worst_cc = cc if _is_point_in_spherical_triangle(cc, p1, p2, p3) else None
            
    return max_radius, worst_i, worst_j, worst_k

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Local Search Optimization with Simulated Annealing
    start_time = time.time()
    time_limit = 118.0 # Leave 2s buffer
    
    current_radius = _covering_radius(points)[0]
    best_radius = current_radius
    best_points = points.copy()
    
    T = 0.05 # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.995
    
    step_size = 0.05
    
    while time.time() - start_time < time_limit:
        # Local Search Phase
        local_start = time.time()
        prev_radius = current_radius
        patience = 0
        max_patience = 20
        
        while time.time() - local_start < 5.0: # Run local search for max 5s or until stuck
            radius, i, j, k = _covering_radius(points)
            current_radius = radius
            
            if radius < best_radius:
                best_radius = radius
                best_points = points.copy()
                
            if radius < prev_radius:
                prev_radius = radius
                patience = 0
            else:
                patience += 1
                if patience > max_patience:
                    break
            
            p1, p2, p3 = points[i], points[j], points[k]
            cc, r_cc = _circumcenter_sphere(p1, p2, p3)
            
            if not _is_point_in_spherical_triangle(cc, p1, p2, p3):
                d12 = _angular_distance(p1, p2)
                d23 = _angular_distance(p2, p3)
                d31 = _angular_distance(p3, p1)
                
                if d12 >= d23 and d12 >= d31:
                    t1 = _normalize(p2 - (np.dot(p1, p2)*p1))
                    t1 = -t1
                    points[i] = _normalize(p1 + step_size * t1)
                    t2 = _normalize(p1 - (np.dot(p2, p1)*p2))
                    t2 = -t2
                    points[j] = _normalize(p2 + step_size * t2)
                elif d23 >= d12 and d23 >= d31:
                    t2 = _normalize(p3 - (np.dot(p2, p3)*p2))
                    t2 = -t2
                    points[j] = _normalize(p2 + step_size * t2)
                    t3 = _normalize(p2 - (np.dot(p3, p2)*p3))
                    t3 = -t3
                    points[k] = _normalize(p3 + step_size * t3)
                else:
                    t3 = _normalize(p1 - (np.dot(p3, p1)*p3))
                    t3 = -t3
                    points[k] = _normalize(p3 + step_size * t3)
                    t1 = _normalize(p3 - (np.dot(p1, p3)*p1))
                    t1 = -t1
                    points[i] = _normalize(p1 + step_size * t1)
            else:
                proj1 = cc - np.dot(cc, p1) * p1
                t1 = _normalize(proj1)
                points[i] = _normalize(p1 + step_size * t1)
                
                proj2 = cc - np.dot(cc, p2) * p2
                t2 = _normalize(proj2)
                points[j] = _normalize(p2 + step_size * t2)
                
                proj3 = cc - np.dot(cc, p3) * p3
                t3 = _normalize(proj3)
                points[k] = _normalize(p3 + step_size * t3)
        
        # Annealing Step: Perturb and restart local search
        if T > T_min:
            # Random perturbation
            noise = np.random.randn(N, 3) * (T * 0.1)
            points = points + noise
            # Normalize points to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0 # Avoid division by zero
            points = points / norms
            
            T *= cooling_rate
            # Decay step size slightly
            step_size = max(0.01, step_size * 0.99)
            
        # If we used all time, break
        if time.time() - start_time > time_limit - 10:
            break
            
    return best_points
# EVOLVE-BLOCK-END
