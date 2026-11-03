"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _compute_covering_radius(points):
    """Compute the covering radius (max circumradius of hull facets)."""
    try:
        hull = ConvexHull(points)
        max_ang = 0.0
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            
            # Ensure normal points outward (away from origin)
            # For a convex hull of points on the unit sphere, 
            # the origin is inside if the points cover the sphere.
            # The normal should point away from the origin.
            if np.dot(normal, v1) < 0:
                normal = -normal
                
            # The circumcenter is in the direction of the normal.
            # The angular distance is arccos(dot(normal, v1))
            # Note: dot(normal, v1) should be the same for v2, v3 if exact.
            d = np.dot(normal, v1)
            d = np.clip(d, -1.0, 1.0)
            ang = np.arccos(d)
            
            if ang > max_ang:
                max_ang = ang
        return max_ang
    except Exception:
        return np.pi

def _optimize(points, time_limit):
    """Local optimization loop."""
    start_time = time.time()
    pts = points.copy()
    current_radius = _compute_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < time_limit:
        # Find worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        worst_facet_idx = -1
        max_ang = -1.0
        worst_normal = None
        worst_simp = None
        
        for idx, simplex in enumerate(hull.simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            if np.dot(normal, v1) < 0:
                normal = -normal
            
            d = np.clip(np.dot(normal, v1), -1.0, 1.0)
            ang = np.arccos(d)
            
            if ang > max_ang:
                max_ang = ang
                worst_facet_idx = idx
                worst_normal = normal
                worst_simp = simplex
        
        if worst_facet_idx == -1:
            break
        
        # Move vertices of worst facet towards circumcenter (normal)
        v1 = pts[worst_simp[0]].copy()
        v2 = pts[worst_simp[1]].copy()
        v3 = pts[worst_simp[2]].copy()
        
        alpha = step_size
        pts[worst_simp[0]] = v1 + alpha * worst_normal
        pts[worst_simp[1]] = v2 + alpha * worst_normal
        pts[worst_simp[2]] = v3 + alpha * worst_normal
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        new_radius = _compute_covering_radius(pts)
        
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
            no_improve_count = 0
            # Can increase step slightly if improving consistently? 
            # step_size = min(step_size * 1.01, 0.1)
        else:
            # Revert
            pts[worst_simp[0]] = v1
            pts[worst_simp[1]] = v2
            pts[worst_simp[2]] = v3
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            
            no_improve_count += 1
            step_size *= 0.95
            if step_size < 1e-5:
                break
            if no_improve_count > max_no_improve:
                break
                
    return best_pts, best_radius

def construct():
    start_time = time.time()
    total_time_limit = 140.0 # Safety margin
    
    # Strategy: Multiple restarts
    # 1. Fibonacci
    # 2. Random (seed 42)
    # 3. Random (seed 123)
    
    candidates = []
    
    # Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_pts)
    
    # Random 1
    rng = np.random.default_rng(42)
    rand_pts1 = rng.normal(size=(N, 3))
    rand_pts1 /= np.linalg.norm(rand_pts1, axis=1, keepdims=True)
    candidates.append(rand_pts1)
    
    # Random 2
    rng = np.random.default_rng(123)
    rand_pts2 = rng.normal(size=(N, 3))
    rand_pts2 /= np.linalg.norm(rand_pts2, axis=1, keepdims=True)
    candidates.append(rand_pts2)
    
    best_pts = fib_pts
    best_radius = np.inf
    
    # Distribute time among candidates
    # If we have 3 candidates, give each ~40s, leaving 20s for final check/buffer.
    # But random points might converge faster or slower.
    # Let's run them sequentially with a time check.
    
    for idx, init_pts in enumerate(candidates):
        time_left = total_time_limit - (time.time() - start_time)
        if time_left < 10: # Need at least 10s to do anything meaningful and return
            break
            
        # Allocate time for this candidate: min(40s, time_left - 5s buffer)
        alloc_time = min(40.0, time_left - 5.0)
        
        if alloc_time <= 0:
            continue
            
        res_pts, res_rad = _optimize(init_pts, alloc_time)
        
        if res_rad < best_radius:
            best_radius = res_rad
            best_pts = res_pts
            
    return best_pts
# EVOLVE-BLOCK-END
