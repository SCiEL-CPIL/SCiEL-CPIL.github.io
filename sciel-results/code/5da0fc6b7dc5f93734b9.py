import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Returns the radius in degrees.
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
        return 180.0
        
    return np.degrees(max_radius)

def _get_facet_data(points, hull):
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

def _optimize(points, deadline, verbose=False):
    """
    Local optimization loop with improved move computation.
    """
    current_radius = _compute_covering_radius(points)
    best_radius = current_radius
    best_points = points.copy()
    
    step_size = 0.03
    min_step_size = 1e-6
    no_improve_count = 0
    max_no_improve = 40
    
    try:
        hull = ConvexHull(points)
        centers, radii, facets = _get_facet_data(points, hull)
    except Exception:
        return best_points, best_radius

    while time.time() < deadline:
        
        # Check if we should restart or kick
        if no_improve_count > max_no_improve:
            # Random kick to escape local minimum
            rng = np.random.default_rng()
            kick_sigma = 0.03 + 0.02 * (no_improve_count - max_no_improve) / 10.0
            kick_sigma = min(kick_sigma, 0.15)
            kick = rng.normal(0, kick_sigma, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            step_size = 0.03
            no_improve_count = 0
            
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_facet_data(points, hull)
            except Exception:
                return best_points, best_radius
            continue

        # Compute move direction: for each point, sum weighted tangent vectors
        # toward circumcenters of all adjacent facets, weighted by radius^2
        max_move = np.zeros_like(points)
        
        # For each facet, distribute its "force" to its three vertices
        # Weight by radius squared to prioritize large holes
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k < 1e-8:
                continue
            
            c_k = centers[k]
            tri = facets[k]
            weight = r_k ** 2
            
            for idx in tri:
                v = c_k - points[idx]
                dot_vp = np.dot(v, points[idx])
                tangent_v = v - dot_vp * points[idx]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[idx] += (tangent_v / norm_tv) * weight
        
        # Normalize move vector
        move_norm = np.linalg.norm(max_move)
        if move_norm > 1e-10:
            max_move = max_move / move_norm
            
            # Line search with backtracking
            alpha = step_size
            improved = False
            best_alpha = 0
            best_new_radius = current_radius
            
            for _ in range(8):
                new_points = points + alpha * max_move
                new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
                
                new_radius = _compute_covering_radius(new_points)
                
                if new_radius < best_new_radius:
                    best_new_radius = new_radius
                    best_alpha = alpha
                    best_new_points = new_points.copy()
                
                if new_radius < current_radius * 0.999:
                    # Accept this step
                    current_radius = new_radius
                    points = new_points.copy()
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    # Increase step size slightly if we are making good progress
                    if new_radius < current_radius * 0.99:
                        step_size = min(step_size * 1.15, 0.08)
                    else:
                        step_size = min(step_size * 1.05, 0.08)
                    break
                else:
                    alpha *= 0.5
            
            if not improved:
                # If we found a better point during line search but not accepted,
                # use it if it's significantly better
                if best_new_radius < current_radius * 0.995 and best_alpha > 0:
                    current_radius = best_new_radius
                    points = best_new_points.copy()
                    improved = True
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # No improvement, decrease step size
                    step_size = max(step_size * 0.7, min_step_size)
                    no_improve_count += 1
                    
                    if step_size <= min_step_size * 1.5:
                        # Force a kick
                        no_improve_count = max_no_improve + 1
        else:
            # No move direction? Stuck
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                pass

        # Recompute hull and facets for next iteration
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_facet_data(points, hull)
        except Exception:
            break

    return best_points, best_radius

def _fibonacci_spiral(rng=None):
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _random_points(rng):
    """Generate random points on sphere."""
    points = rng.normal(0, 1, (N, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _optimize_with_annealing(points, deadline, verbose=False):
    """
    Simulated annealing style optimization.
    """
    current_radius = _compute_covering_radius(points)
    best_radius = current_radius
    best_points = points.copy()
    
    temperature = 0.05
    min_temperature = 1e-6
    cooling_rate = 0.95
    no_improve_count = 0
    max_no_improve = 50
    
    try:
        hull = ConvexHull(points)
        centers, radii, facets = _get_facet_data(points, hull)
    except Exception:
        return best_points, best_radius

    while time.time() < deadline and temperature > min_temperature:
        
        if no_improve_count > max_no_improve:
            # Kick
            rng = np.random.default_rng()
            kick_sigma = temperature * 2
            kick = rng.normal(0, kick_sigma, points.shape)
            points += kick
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            current_radius = _compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            no_improve_count = 0
            temperature = max(temperature, 0.03)
            
            try:
                hull = ConvexHull(points)
                centers, radii, facets = _get_facet_data(points, hull)
            except Exception:
                return best_points, best_radius
            continue

        # Compute move direction
        max_move = np.zeros_like(points)
        
        for k in range(len(facets)):
            r_k = radii[k]
            if r_k < 1e-8:
                continue
            
            c_k = centers[k]
            tri = facets[k]
            weight = r_k ** 2
            
            for idx in tri:
                v = c_k - points[idx]
                dot_vp = np.dot(v, points[idx])
                tangent_v = v - dot_vp * points[idx]
                norm_tv = np.linalg.norm(tangent_v)
                if norm_tv > 1e-12:
                    max_move[idx] += (tangent_v / norm_tv) * weight
        
        move_norm = np.linalg.norm(max_move)
        if move_norm > 1e-10:
            max_move = max_move / move_norm
            
            # Add noise
            rng = np.random.default_rng()
            noise = rng.normal(0, temperature, max_move.shape)
            move = max_move + noise
            move /= np.linalg.norm(move)
            
            # Try the move
            alpha = 0.02
            new_points = points + alpha * move
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            
            # Metropolis acceptance
            if new_radius < current_radius:
                points = new_points
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Accept with probability exp(-(new_radius - current_radius) / temperature)
                delta = new_radius - current_radius
                if delta > 0:
                    accept_prob = np.exp(-delta / (temperature * 0.1))
                    if rng.random() < accept_prob:
                        points = new_points
                        current_radius = new_radius
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
                else:
                    points = new_points
                    current_radius = new_radius
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
            
            temperature *= cooling_rate
        else:
            no_improve_count += 1
            temperature *= cooling_rate

        # Recompute hull
        try:
            hull = ConvexHull(points)
            centers, radii, facets = _get_facet_data(points, hull)
        except Exception:
            break

    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_radius = float('inf')
    
    strategies = [
        ('fibonacci', _fibonacci_spiral),
        ('random', _random_points),
        ('fibonacci_rotated', None),
    ]
    
    run_id = 0
    base_seed = 42
    
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        rng = np.random.default_rng(base_seed + run_id)
        
        # Choose initialization strategy
        if run_id % 3 == 0:
            # Fibonacci spiral with perturbation
            points = _fibonacci_spiral()
            points += rng.normal(0, 0.02, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        elif run_id % 3 == 1:
            # Random points
            points = _random_points(rng)
        else:
            # Fibonacci with different rotation
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + rng.uniform(0, 1))
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            points += rng.normal(0, 0.015, points.shape)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Choose optimization method
        if run_id % 2 == 0:
            method = _optimize
        else:
            method = _optimize_with_annealing
        
        # Allocate time for this run
        run_deadline = time.time() + remaining * 0.75
        
        res_points, res_radius = method(points, run_deadline)
        
        if res_radius < best_radius:
            best_radius = res_radius
            best_points = res_points
            
        run_id += 1
        
        # If we found a very good result, stop early
        if best_radius < 11.0:
            break
    
    return best_points
