import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the normal to the plane defined by p1, p2, p3
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                continue
                
            c = cross / norm_cross
            
            # Ensure the normal points outward (away from origin)
            if np.dot(c, p1) < 0:
                c = -c
            
            # The angular distance from the circumcenter c to the vertices is the circumradius.
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        # If hull fails, return a large value
        return np.pi / 2
        
    return np.degrees(max_radius)

def _fibonacci_spiral(N):
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _gradient_descent(points, time_limit=100.0):
    """
    Perform gradient-based optimization using the repulsive force model.
    This avoids expensive ConvexHull calls during the main loop.
    We minimize a smooth approximation of the covering radius by maximizing
    the minimum distance between points (or minimizing variance of distances).
    Here we use a repulsive potential to spread points out.
    """
    start_time = time.time()
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = _compute_covering_radius(current_points)
    
    # Parameters
    step_size = 0.05
    decay = 0.995
    min_step = 1e-5
    temperature = 1.0
    cooling = 0.99
    
    while time.time() < start_time + time_limit:
        # Compute pairwise distances
        # G = points @ points.T
        G = np.dot(current_points, current_points.T)
        G = np.clip(G, -1.0, 1.0)
        
        # We want to maximize the minimum distance.
        # A common proxy is to minimize the sum of exp(-k * dist) or similar.
        # Or simply move points away from their nearest neighbors.
        
        # For each point, find its nearest neighbor and move away from it.
        # This is a simple repulsive force.
        # dists is the distance matrix (angular)
        dists = np.arccos(G)
        np.fill_diagonal(dists, np.inf)
        
        nearest_idx = np.argmin(dists, axis=1)
        min_dists = np.min(dists, axis=1)
        
        # The "worst" points are those with the smallest minimum distance to neighbors?
        # No, for covering radius, we care about the largest empty cap.
        # However, a well-distributed set has uniform nearest-neighbor distances.
        # Minimizing the variance of nearest-neighbor distances helps.
        
        # Let's use a potential energy approach:
        # E = sum_{i != j} exp(-alpha * dist(i, j))
        # Gradient of this w.r.t. point i pushes it away from neighbors.
        
        alpha = 5.0
        # Compute forces
        forces = np.zeros_like(current_points)
        
        # Vectorized force calculation is complex, let's loop for clarity and speed is okay for N=130
        for i in range(N):
            f = np.zeros(3)
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                # Force magnitude proportional to exp(-alpha * d)
                mag = np.exp(-alpha * d)
                # Direction is away from j
                # Vector from i to j is points[j] - points[i]
                # Project onto tangent plane at i
                v = current_points[j] - current_points[i]
                # Remove radial component
                radial = np.dot(v, current_points[i]) * current_points[i]
                tangential = v - radial
                norm_t = np.linalg.norm(tangential)
                if norm_t > 1e-12:
                    tangential /= norm_t
                else:
                    continue
                f -= mag * tangential # Move away, so negative direction of v?
                # Wait, v points i->j. To move away, we go in -v direction.
                # So force is -mag * tangential. Correct.
            
            forces[i] = f
        
        # Normalize forces to step size
        for i in range(N):
            norm_f = np.linalg.norm(forces[i])
            if norm_f > 1e-12:
                forces[i] /= norm_f
        
        # Update points
        new_points = current_points.copy()
        for i in range(N):
            new_points[i] = current_points[i] + step_size * forces[i]
            new_points[i] /= np.linalg.norm(new_points[i])
        
        # Simulated annealing acceptance
        new_radius = _compute_covering_radius(new_points)
        
        if new_radius < best_radius or np.random.random() < np.exp(-(new_radius - best_radius) / (temperature * 0.1)):
            current_points = new_points
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
        
        # Decay parameters
        step_size *= decay
        temperature *= cooling
        
        if step_size < min_step:
            break

    return best_points, best_radius

def construct():
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin before 150s limit
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Phase 1: Gradient Descent with Repulsion
    # This quickly improves the distribution by spreading points out.
    points, radius = _gradient_descent(points, time_limit=60.0)
    if radius < best_radius:
        best_radius = radius
        best_points = points.copy()
    
    # Phase 2: Local Refinement using Convex Hull
    # Now that points are well-distributed, we can afford to use ConvexHull
    # to precisely identify the worst facets and correct them.
    
    if time.time() < deadline:
        step_size = 0.02
        min_step_size = 1e-4
        current_points = best_points.copy()
        
        while time.time() < deadline:
            # Compute hull and find worst facet
            try:
                hull = ConvexHull(current_points)
                facets = hull.simplices
                
                worst_facet_idx = -1
                worst_radius = 0.0
                worst_circumcenter = None
                worst_vertices = None
                
                for idx, tri in enumerate(facets):
                    p1 = current_points[tri[0]]
                    p2 = current_points[tri[1]]
                    p3 = current_points[tri[2]]
                    
                    v1 = p2 - p1
                    v2 = p3 - p1
                    cross = np.cross(v1, v2)
                    norm_cross = np.linalg.norm(cross)
                    
                    if norm_cross < 1e-12:
                        continue
                        
                    c = cross / norm_cross
                    
                    # Ensure normal points outward
                    if np.dot(c, p1) < 0:
                        c = -c
                    
                    dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                    dist = np.arccos(dot_val)
                    
                    if dist > worst_radius:
                        worst_radius = dist
                        worst_facet_idx = idx
                        worst_circumcenter = c
                        worst_vertices = tri
                        
            except Exception:
                break
                
            if worst_facet_idx == -1:
                break
                
            # Move vertices towards the circumcenter
            prev_v1 = current_points[worst_vertices[0]].copy()
            prev_v2 = current_points[worst_vertices[1]].copy()
            prev_v3 = current_points[worst_vertices[2]].copy()
            
            for v_idx in worst_vertices:
                p = current_points[v_idx]
                direction = worst_circumcenter - p
                norm_dir = np.linalg.norm(direction)
                if norm_dir < 1e-12:
                    continue
                direction /= norm_dir
                
                current_points[v_idx] = current_points[v_idx] + step_size * direction
                current_points[v_idx] /= np.linalg.norm(current_points[v_idx])
            
            # Check improvement
            current_radius = _compute_covering_radius(current_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = current_points.copy()
                step_size *= 0.98
            else:
                # Revert if worse
                current_points[worst_vertices[0]] = prev_v1
                current_points[worst_vertices[1]] = prev_v2
                current_points[worst_vertices[2]] = prev_v3
                
                # Reduce step size
                step_size *= 0.9
                
                if step_size < min_step_size:
                    # Try random perturbation to escape local minima
                    for v_idx in worst_vertices:
                        noise = np.random.normal(0, 0.01, 3)
                        current_points[v_idx] = current_points[v_idx] + noise
                        current_points[v_idx] /= np.linalg.norm(current_points[v_idx])
                    
                    step_size = 0.01
                    current_radius = _compute_covering_radius(current_points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = current_points.copy()
                    else:
                        # Revert jitter
                        current_points[worst_vertices[0]] = prev_v1
                        current_points[worst_vertices[1]] = prev_v2
                        current_points[worst_vertices[2]] = prev_v3

    return best_points
