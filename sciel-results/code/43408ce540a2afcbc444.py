import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius_vectorized(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Vectorized version for speed.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Get points for all facets: (M, 3, 3)
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p1
        
        # Cross products
        cross = np.cross(v1, v2)
        norms = np.linalg.norm(cross, axis=1)
        
        # Mask for degenerate facets
        valid = norms > 1e-12
        
        if not np.any(valid):
            return 180.0
            
        # Normalize
        c = cross[valid] / norms[valid, np.newaxis]
        
        # Centroids
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        
        # Check orientation: c should point towards centroid
        dots = np.sum(c * centroids, axis=1)
        flip_mask = dots < 0
        c[flip_mask] = -c[flip_mask]
        
        # Radii
        # dot(c, p1)
        dots_p = np.sum(c * p1[valid], axis=1)
        dots_p = np.clip(dots_p, -1.0, 1.0)
        radii = np.arccos(dots_p)
        
        # The covering radius is the max of these
        # Note: We must ensure we are taking the radius of the "empty cap".
        # For a convex hull of points on the sphere, the normal points outwards.
        # The circumcenter on the sphere is either c or -c.
        # The one that is "inside" the spherical triangle is the one closer to the centroid.
        # So the distance is arccos(dot(c, p1)).
        
        return np.degrees(np.max(radii))
        
    except Exception:
        return 180.0

def _get_circumcenters_and_radii_vectorized(points, hull):
    """
    Compute circumcenters and radii for all hull facets.
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    cross = np.cross(v1, v2)
    norms = np.linalg.norm(cross, axis=1)
    
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    valid = norms > 1e-12
    
    if not np.any(valid):
        return centers, radii, facets

    c = cross[valid] / norms[valid, np.newaxis]
    
    centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
    
    dots = np.sum(c * centroids, axis=1)
    flip_mask = dots < 0
    c[flip_mask] = -c[flip_mask]
    
    centers[valid] = c
    
    dots_p = np.sum(c * p1[valid], axis=1)
    dots_p = np.clip(dots_p, -1.0, 1.0)
    radii[valid] = np.arccos(dots_p)
    
    # For invalid facets, radius is 0, center is p1 (arbitrary)
    # But we don't really care about them for optimization
    
    return centers, radii, facets

def _optimize(points, deadline):
    """
    Local optimization loop using vectorized calculations.
    """
    current_radius = _compute_covering_radius_vectorized(points)
    best_radius = current_radius
    best_points = points.copy()
    
    step_size = 0.05
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 20
    
    try:
        hull = ConvexHull(points)
        centers, radii, facets = _get_circumcenters_and_radii_vectorized(points, hull)
    except Exception:
        return best_points, best_radius

    while time.time() < deadline:
        
        if no_improve_count > max_no_improve:
            # Random kick
            rng = np.random.default_rng()
            kick = rng.normal(0, 0.05, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            current_radius = _compute_covering_radius_vectorized(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            step_size = 0.05
            no_improve_count = 0
            
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_circumcenters_and_radii_vectorized(points, hull)
            except Exception:
                return best_points, best_radius
            continue

        # Compute move direction
        # For each point, find the adjacent facet with the max radius
        # and move towards its center.
        
        max_move = np.zeros_like(points)
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        # Vectorized assignment is hard here because we need "max radius per point"
        # but the facets are sparse.
        # Loop over facets (M ~ 260) is fast.
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k <= 0: continue
            
            c_k = centers[k]
            tri = facets[k]
            
            # Check 3 vertices
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    # Weight by radius to prioritize large holes
                    weight = point_max_radius[i] ** 2
                    max_move[i] = (tangent_v / norm_tv) * weight
        
        move_norm = np.linalg.norm(max_move)
        if move_norm > 1e-10:
            max_move = max_move / move_norm
            
            # Line search
            alpha = step_size
            improved = False
            for _ in range(5):
                new_points = points + alpha * max_move
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_radius = _compute_covering_radius_vectorized(new_points)
                
                if new_radius < current_radius:
                    current_radius = new_radius
                    points = new_points
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    step_size = min(step_size * 1.1, 0.1)
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                step_size = max(step_size * 0.5, min_step_size)
                no_improve_count += 1
                if step_size <= min_step_size:
                    no_improve_count = max_no_improve + 1
        else:
            no_improve_count += 1

        # Recompute hull
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii_vectorized(points, hull)
        except Exception:
            break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_radius = float('inf')
    
    base_seed = 42
    run_id = 0
    
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5.0: # Not enough time for a full run
            break
            
        rng = np.random.default_rng(base_seed + run_id)
        
        # Fibonacci spiral initialization
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Perturb
        points += rng.normal(0, 0.02, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize
        run_deadline = time.time() + remaining * 0.8
        res_points, res_radius = _optimize(points, run_deadline)
        
        if res_radius < best_radius:
            best_radius = res_radius
            best_points = res_points
            
        run_id += 1
        
        if best_radius < 11.32: # Close to HSS
             # We might stop early if we are very close, but let's try to improve further
             if best_radius < 11.0:
                 break

    return best_points
