import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _covering_radius(points):
    """Compute the covering radius in radians."""
    try:
        hull = ConvexHull(points)
        max_rad = 0.0
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Circumcenter of spherical triangle
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            
            # Choose the correct sign
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            if rad > max_rad:
                max_rad = rad
    except Exception:
        return np.pi
    return max_rad

def _smooth_max(x, beta=10.0):
    """Smooth maximum function."""
    if len(x) == 0:
        return 0.0
    max_x = np.max(x)
    if max_x < -100:
        return max_x
    return max_x + np.log(np.sum(np.exp((x - max_x) * beta)) / len(x)) / beta

def _objective_flat(flat_points):
    """Objective function for optimization: covering radius + regularization."""
    points = flat_points.reshape(N, 3)
    
    # Ensure points are on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    points = points / norms
    
    # Check for coincident points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.arccos(np.clip(np.dot(points[i], points[j]), -1.0, 1.0))
            if d < min_dist:
                min_dist = d
    
    # Penalty for points too close together
    penalty = 0.0
    if min_dist < 0.01:
        penalty = (0.01 - min_dist) ** 2 * 100
    
    # Compute covering radius
    rad = _covering_radius(points)
    
    # Also compute a smooth surrogate
    # For each point, compute distance to nearest other point
    # This is expensive, so we use the hull-based radius
    
    return rad + penalty

def construct():
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    max_time = 140.0  # Safety margin before 150s kill
    
    # Phase 1: Greedy Lloyd-style optimization
    step_size = 0.03
    for it in range(3000):
        if time.time() - start_time > max_time:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_facet_rad = 0.0
        worst_circumcenter = None
        worst_indices = None
        
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            n = np.cross(v1 - v2, v1 - v3)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            centroid = (v1 + v2 + v3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
            rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
            
            if rad > max_facet_rad:
                max_facet_rad = rad
                worst_circumcenter = c
                worst_indices = simplex
        
        if worst_indices is None:
            break
        
        # Move vertices of worst facet toward circumcenter
        for j in range(3):
            v_idx = worst_indices[j]
            v = points[v_idx]
            new_v = v + step_size * (worst_circumcenter - v)
            norm = np.linalg.norm(new_v)
            if norm > 1e-10:
                points[v_idx] = new_v / norm
        
        # Also apply small random perturbations to escape local minima
        if it % 50 == 0:
            for _ in range(3):
                idx = np.random.randint(N)
                # Random direction on tangent plane
                tangent = np.random.randn(3)
                tangent -= np.dot(tangent, points[idx]) * points[idx]
                norm_t = np.linalg.norm(tangent)
                if norm_t > 1e-10:
                    tangent /= norm_t
                    new_v = points[idx] + 0.01 * tangent
                    norm = np.linalg.norm(new_v)
                    if norm > 1e-10:
                        points[idx] = new_v / norm
        
        current_rad = _covering_radius(points)
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
    
    # Phase 2: Nelder-Mead polishing on the worst few facets
    if time.time() - start_time < max_time:
        # Use the best points so far
        points = best_points.copy()
        
        for polish_iter in range(50):
            if time.time() - start_time > max_time:
                break
            
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            
            # Get all facet radii
            facet_rads = []
            facet_data = []
            for simplex in hull.simplices:
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                facet_rads.append(rad)
                facet_data.append((simplex, c))
            
            if not facet_rads:
                break
            
            # Sort by radius descending, take top 3
            sorted_facets = sorted(zip(facet_rads, facet_data), key=lambda x: -x[0])
            top_facets = sorted_facets[:3]
            
            improved = False
            for rad_val, (simplex, circumcenter) in top_facets:
                if time.time() - start_time > max_time:
                    break
                
                # Try small moves of each vertex
                for j in range(3):
                    v_idx = simplex[j]
                    v = points[v_idx]
                    
                    # Try moving toward circumcenter
                    for step in [0.01, 0.005, 0.002]:
                        new_v = v + step * (circumcenter - v)
                        norm = np.linalg.norm(new_v)
                        if norm < 1e-10:
                            continue
                        new_v = new_v / norm
                        
                        old_v = points[v_idx].copy()
                        points[v_idx] = new_v
                        
                        new_rad = _covering_radius(points)
                        if new_rad < best_rad:
                            best_rad = new_rad
                            best_points = points.copy()
                            improved = True
                        else:
                            points[v_idx] = old_v
                    
                    # Try random perturbations
                    for _ in range(5):
                        tangent = np.random.randn(3)
                        tangent -= np.dot(tangent, v) * v
                        norm_t = np.linalg.norm(tangent)
                        if norm_t < 1e-10:
                            continue
                        tangent /= norm_t
                        
                        for step in [0.01, 0.005]:
                            new_v = v + step * tangent
                            norm = np.linalg.norm(new_v)
                            if norm < 1e-10:
                                continue
                            new_v = new_v / norm
                            
                            old_v = points[v_idx].copy()
                            points[v_idx] = new_v
                            
                            new_rad = _covering_radius(points)
                            if new_rad < best_rad:
                                best_rad = new_rad
                                best_points = points.copy()
                                improved = True
                            else:
                                points[v_idx] = old_v
            
            if not improved:
                # Try a larger random restart on a random subset
                for _ in range(10):
                    if time.time() - start_time > max_time:
                        break
                    idx = np.random.randint(N)
                    tangent = np.random.randn(3)
                    tangent -= np.dot(tangent, points[idx]) * points[idx]
                    norm_t = np.linalg.norm(tangent)
                    if norm_t < 1e-10:
                        continue
                    tangent /= norm_t
                    
                    for step in [0.05, 0.02]:
                        new_v = points[idx] + step * tangent
                        norm = np.linalg.norm(new_v)
                        if norm < 1e-10:
                            continue
                        new_v = new_v / norm
                        
                        old_v = points[idx].copy()
                        points[idx] = new_v
                        
                        new_rad = _covering_radius(points)
                        if new_rad < best_rad:
                            best_rad = new_rad
                            best_points = points.copy()
                        else:
                            points[idx] = old_v

    return best_points
