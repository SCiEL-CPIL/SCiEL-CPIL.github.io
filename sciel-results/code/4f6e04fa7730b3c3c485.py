import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure c points to the empty cap side (away from origin relative to triangle centroid)
            centroid = (p1 + p2 + p3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        # If hull fails, return a large value
        return 180.0
        
    return np.degrees(max_radius)

def _get_circumcenters_and_radii(points, hull):
    """
    Compute circumcenters and radii for all hull facets.
    Returns centers (M, 3), radii (M,), and facets (M, 3).
    """
    facets = hull.simplices
    M = len(facets)
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    for k in range(M):
        tri = facets[k]
        p1 = points[tri[0]]
        p2 = points[tri[1]]
        p3 = points[tri[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            centers[k] = p1
            radii[k] = 0.0
            continue
            
        c = cross / norm_cross
        
        centroid = (p1 + p2 + p3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
            
        centers[k] = c
        dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
        dist = np.arccos(dot_val)
        
        radii[k] = dist
        
    return centers, radii, facets

def _optimize(points, deadline):
    """
    Local optimization loop.
    """
    current_radius = _compute_covering_radius(points)
    best_radius = current_radius
    best_points = points.copy()
    
    step_size = 0.05
    min_step_size = 1e-5
    no_improve_count = 0
    max_no_improve = 30
    
    try:
        hull = ConvexHull(points)
        centers, radii, facets = _get_circumcenters_and_radii(points, hull)
    except Exception:
        return best_points, best_radius

    while time.time() < deadline:
        
        # Check if we should restart or kick
        if no_improve_count > max_no_improve:
            # Random kick to escape local minimum
            rng = np.random.default_rng()
            kick = rng.normal(0, 0.05, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            step_size = 0.05
            no_improve_count = 0
            
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_circumcenters_and_radii(points, hull)
            except Exception:
                return best_points, best_radius
            continue

        # Compute gradient/move direction
        # For each point, sum weighted vectors towards circumcenters of adjacent facets
        max_move = np.zeros_like(points)
        
        # Map point index to list of adjacent facets is implicit in the loop
        # We accumulate the move for each point
        point_max_radius = np.zeros(N)
        point_max_center = np.zeros((N, 3))
        
        # First pass: find the dominant facet for each point (largest radius)
        # This is a heuristic: move towards the center of the largest hole touching this point
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k <= 0: continue
            
            c_k = centers[k]
            tri = facets[k]
            for idx in tri:
                if r_k > point_max_radius[idx]:
                    point_max_radius[idx] = r_k
                    point_max_center[idx] = c_k
        
        # Second pass: compute move
        for i in range(N):
            if point_max_radius[i] > 0:
                v = point_max_center[i] - points[i]
                dot_vp = np.dot(v, points[i])
                tangent_v = v - dot_vp * points[i]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    # Weight by radius squared to prioritize large holes
                    weight = point_max_radius[i] ** 2
                    max_move[i] = (tangent_v / norm_tv) * weight
        
        # Normalize move vector to unit length if not zero, then scale by step_size
        move_norm = np.linalg.norm(max_move)
        if move_norm > 1e-10:
            max_move = max_move / move_norm
            
            # Line search
            alpha = step_size
            improved = False
            for _ in range(5):
                new_points = points + alpha * max_move
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_radius = _compute_covering_radius(new_points)
                
                if new_radius < current_radius:
                    # Accept
                    current_radius = new_radius
                    points = new_points
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    # Increase step size slightly if we are making progress
                    step_size = min(step_size * 1.1, 0.1)
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                # If even small step didn't improve, decrease step size and count no-improve
                step_size = max(step_size * 0.5, min_step_size)
                no_improve_count += 1
                
                # If step size is too small, we might be stuck
                if step_size <= min_step_size:
                    # Force a kick
                    no_improve_count = max_no_improve + 1
        else:
            # No move direction? Stuck
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Trigger kick in next iteration
                pass

        # Recompute hull and facets for next iteration
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_circumcenters_and_radii(points, hull)
        except Exception:
            # If hull fails, we are in a bad state, break or kick
            break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_radius = float('inf')
    
    # Use a fixed seed for reproducibility of the first run,
    # but randomize subsequent runs if we have time
    base_seed = 42
    run_id = 0
    
    while time.time() < deadline:
        # Check if we have enough time for another run
        # Estimate time per run: ~20-50s? 
        # Let's just run until deadline, but check inside _optimize too.
        
        rng = np.random.default_rng(base_seed + run_id)
        
        # Initialize with Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Add small random perturbation to break symmetry
        points += rng.normal(0, 0.02, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimize
        # Allocate 80% of remaining time to this run
        remaining = deadline - time.time()
        run_deadline = time.time() + remaining * 0.8
        
        res_points, res_radius = _optimize(points, run_deadline)
        
        if res_radius < best_radius:
            best_radius = res_radius
            best_points = res_points
            
        run_id += 1
        
        # If we found a very good result, we might stop early?
        # The HSS radius is ~11.3 deg.
        # If we get below 10 deg, it's excellent.
        if best_radius < 10.0:
            break

    return best_points
