"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees for points on the unit sphere.
    Uses vectorized circumradius computation for triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        max_circumradius = 0.0
        
        # Handle triangular facets (most common in 3D convex hull)
        tri_facets = [f for f in facets if len(f) == 3]
        
        if len(tri_facets) > 0:
            tri_facets = np.array(tri_facets)
            a = points[tri_facets[:, 0]]
            b = points[tri_facets[:, 1]]
            c = points[tri_facets[:, 2]]
            
            # Compute cross products
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            
            # Triple product: a . (b x c)
            triple = np.einsum('ij,ij->i', a, np.cross(b, c))
            denom = 4.0 * np.abs(triple)
            
            # Avoid division by zero
            valid = denom > 1e-15
            if np.any(valid):
                norms_ab = np.linalg.norm(ab[valid], axis=1)
                norms_bc = np.linalg.norm(bc[valid], axis=1)
                norms_ca = np.linalg.norm(ca[valid], axis=1)
                
                sin_R = (norms_ab * norms_bc * norms_ca) / denom[valid]
                sin_R = np.clip(sin_R, 0.0, 1.0)
                R = np.arcsin(sin_R)
                max_circumradius = max(max_circumradius, np.max(R))
        
        # Handle non-triangular facets (rare, but possible)
        non_tri_facets = [f for f in facets if len(f) != 3]
        for simplex in non_tri_facets:
            pts = points[simplex]
            n_pts = len(pts)
            if n_pts < 3:
                continue
            if n_pts == 4:
                a, b, c, d = pts[0], pts[1], pts[2], pts[3]
                v1 = a - b
                v2 = a - c
                v3 = a - d
                x = np.cross(v1, np.cross(v2, v3))
                nx = np.linalg.norm(x)
                if nx < 1e-15:
                    continue
                x = x / nx
                cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
                R = np.arccos(cos_R)
                max_circumradius = max(max_circumradius, R)
            else:
                centroid = np.mean(pts, axis=0)
                nc = np.linalg.norm(centroid)
                if nc < 1e-15:
                    continue
                centroid = centroid / nc
                for p in pts:
                    cos_R = np.clip(np.dot(centroid, p), -1.0, 1.0)
                    R = np.arccos(cos_R)
                    max_circumradius = max(max_circumradius, R)
        
        return np.degrees(max_circumradius)
    except Exception:
        return np.inf

def _soft_max_energy(points, sigma=0.1):
    """
    Compute a smooth energy function that approximates the covering radius.
    Uses a sum of Gaussian kernels to create a smooth landscape.
    """
    # Compute pairwise distances
    # dists[i, j] = distance between points[i] and points[j]
    dots = points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # For each point, find the max distance to any other point? No, that's packing.
    # For covering, we want to minimize the max distance from any point on sphere to nearest point.
    # A proxy is to minimize the integral of exp(-d^2/sigma^2) over the sphere.
    # Approximating the integral by summing over a grid is too slow.
    # Alternative: Minimize the sum of exp(-d_ij^2 / sigma^2) for all pairs? That's packing (maximize min dist).
    # Covering is dual. 
    # Let's use a different proxy: Minimize the max of the "Voronoi" radius.
    # Actually, a common trick for covering is to minimize the sum of squares of distances to a fixed grid,
    # or to maximize the minimum distance to a grid.
    
    # Let's try: Minimize E = sum_i max_j d(i, j)^2 ? No, that's packing.
    # Covering radius R is the smallest R such that the union of caps of radius R covers the sphere.
    # Let's use a grid-based approximation for the energy to guide the global optimization.
    # Generate a fixed grid of points on the sphere.
    # E = max_{y in grid} (min_{x in points} d(x, y))^2
    
    # This is expensive to compute and differentiate.
    # Let's stick to the convex hull method but use a smarter initial guess.
    
    # Actually, the prompt suggests spectral optimization. 
    # Let's implement a simple gradient descent on the convex hull circumradii.
    # We can compute the gradient of the max circumradius with respect to point positions.
    
    return 0.0 # Placeholder

def _refine_local(points, max_time=5.0):
    """
    Refine points using gradient descent on the convex hull circumradii.
    """
    start_time = time.time()
    current_points = points.copy()
    current_radius = _covering_radius_fast(current_points)
    
    # We will use a simple coordinate descent with line search on the worst facet.
    # This is similar to the current program but we can make it more robust.
    
    # To save time, we'll just do a few iterations of moving points towards the circumcenter of the worst facet.
    for _ in range(20):
        if time.time() - start_time > max_time:
            break
        
        # Get worst facet
        try:
            hull = ConvexHull(current_points)
            facets = hull.simplices
            
            max_r = -1
            worst_facet = None
            worst_center = None
            
            for f in facets:
                if len(f) != 3:
                    continue
                a, b, c = current_points[f[0]], current_points[f[1]], current_points[f[2]]
                v1 = a - b
                v2 = a - c
                v = np.cross(v1, v2)
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-10:
                    continue
                x = v / norm_v
                cos_r = np.clip(np.dot(x, a), -1.0, 1.0)
                r = np.arccos(cos_r)
                if r > max_r:
                    max_r = r
                    worst_facet = f
                    worst_center = x
            
            if worst_facet is None:
                break
                
            # Move vertices of the worst facet towards the center?
            # Or move them in a direction that reduces the radius.
            # The radius is reduced if the vertices move away from the center?
            # No, the center is the furthest point. Moving vertices towards the center reduces the triangle size?
            # Actually, moving vertices towards the center of the sphere (origin) is not allowed (must stay on surface).
            # Moving vertices along the surface towards each other reduces the triangle area and thus the circumradius.
            
            # Let's try moving each vertex of the worst facet slightly towards the other two vertices.
            improved = False
            for i in worst_facet:
                p_i = current_points[i]
                # Direction towards centroid of the other two
                others = [j for j in worst_facet if j != i]
                centroid = (current_points[others[0]] + current_points[others[1]]) / 2.0
                centroid /= np.linalg.norm(centroid)
                
                # Try small steps
                for step in [0.01, 0.005, 0.001]:
                    new_p = p_i + step * centroid
                    new_p /= np.linalg.norm(new_p)
                    
                    candidate = current_points.copy()
                    candidate[i] = new_p
                    
                    r = _covering_radius_fast(candidate)
                    if r < current_radius:
                        current_points = candidate
                        current_radius = r
                        improved = True
                        break
            
            if not improved:
                break
                
        except Exception:
            break
            
    return current_points, current_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate multiple diverse initial configurations
    # Strategy: Use different random seeds and different initialization methods
    # Method A: Random uniform
    # Method B: Fibonacci spiral (deterministic, good spread)
    # Method C: Grid based
    
    candidates = []
    
    # Method B: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_points)
    
    # Method A: Random uniform with different seeds
    np.random.seed(42)
    for seed in range(5):
        if time.time() > deadline - 20.0:
            break
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        candidates.append(rand_points)
        
    best_points = None
    best_radius = np.inf
    
    # 2. Refine each candidate
    for points in candidates:
        if time.time() > deadline - 10.0:
            break
            
        # Quick check: if points are too clustered, skip?
        # Just refine
        refined, radius = _refine_local(points, max_time=20.0)
        
        if radius < best_radius:
            best_radius = radius
            best_points = refined
            
    # 3. Final polish on the best candidate
    if best_points is not None and time.time() < deadline - 5.0:
        final_points, final_radius = _refine_local(best_points, max_time=30.0)
        if final_radius < best_radius:
            best_points = final_points
            best_radius = final_radius
            
    return best_points
# EVOLVE-BLOCK-END
