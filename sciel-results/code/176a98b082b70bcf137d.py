import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _facet_circumradii(points):
    """Compute circumradii of all facets of the convex hull."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None
    
    radii = []
    circumcenters = []
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
        radii.append(rad)
        circumcenters.append(c)
    
    if not radii:
        return None, None
    return np.array(radii), np.array(circumcenters)

def _covering_radius(points):
    """Compute the covering radius in radians."""
    radii, _ = _facet_circumradii(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _smooth_objective(points, beta=100.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and allows gradient-based optimization.
    """
    radii, _ = _facet_circumradii(points)
    if radii is None:
        return np.pi, None
    
    # log-sum-exp for numerical stability
    max_r = np.max(radii)
    if max_r < 1e-10:
        return max_r, None
    
    # Soft max: (1/beta) * log(sum(exp(beta * r_i)))
    # For large beta, this approximates max(r_i)
    scaled = beta * (radii - max_r)
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled))) + max_r
    soft_max = log_sum / beta
    
    # Compute gradient w.r.t. points
    # d(soft_max)/d(r_i) = exp(beta * r_i) / sum(exp(beta * r_j))
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # Now compute gradient of each r_i w.r.t. points
    # r_i = arccos(c_i . v1_i) where c_i is circumcenter of facet i
    # This is complex, so we'll use finite differences or a simpler approach
    
    return soft_max, weights

def _surrogate_gradient(points, beta=100.0):
    """
    Compute gradient of the smooth surrogate objective w.r.t. points.
    Uses finite differences for simplicity and robustness.
    """
    radii, circumcenters = _facet_circumradii(points)
    if radii is None or len(radii) == 0:
        return None
    
    # Compute soft max weights
    max_r = np.max(radii)
    if max_r < 1e-10:
        return np.zeros_like(points)
    
    scaled = beta * (radii - max_r)
    scaled = np.clip(scaled, -50, 50)
    weights = np.exp(scaled) / np.sum(np.exp(scaled))
    
    # Gradient of soft_max w.r.t. each point
    grad = np.zeros_like(points)
    
    # For each facet, compute contribution to gradient
    # r_i = arccos(c_i . v_{i,0})
    # d(r_i)/d(v_{i,0}) = -c_i / sqrt(1 - (c_i . v_{i,0})^2)
    # d(soft_max)/d(v_{i,0}) = weights[i] * d(r_i)/d(v_{i,0})
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    
    for idx, simplex in enumerate(hull.simplices):
        if idx >= len(radii):
            break
        w = weights[idx]
        if w < 1e-10:
            continue
        
        v1 = points[simplex[0]]
        c = circumcenters[idx]
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0 + 1e-10, 1.0 - 1e-10)
        denom = np.sqrt(1.0 - dot * dot)
        
        if denom < 1e-10:
            continue
        
        # Gradient of arccos(dot) w.r.t. v1 is -c / denom
        # But we need to account for the fact that v1 is on the sphere
        # Project gradient to tangent space: grad_tangent = grad - (grad . v1) * v1
        grad_v1 = -c / denom
        grad_tangent = grad_v1 - np.dot(grad_v1, v1) * v1
        
        grad[simplex[0]] += w * grad_tangent
    
    return grad

def _gradient_step(points, grad, step_size):
    """
    Perform a gradient descent step on the sphere.
    Uses the Riemannian gradient: project to tangent space, then exponential map.
    """
    # Project gradient to tangent space at each point
    grad_tangent = np.zeros_like(points)
    for i in range(len(points)):
        v = points[i]
        g = grad[i]
        # Tangent projection: g - (g . v) * v
        grad_tangent[i] = g - np.dot(g, v) * v
    
    # Move along negative gradient
    new_points = points - step_size * grad_tangent
    
    # Normalize to unit sphere
    new_points = _normalize(new_points)
    
    return new_points

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
    max_time = 140.0  # Safety margin before 150s limit
    
    # Phase 1: Gradient descent with smooth surrogate
    step_size = 0.01
    momentum = 0.9
    velocity = np.zeros_like(points)
    fail_count = 0
    
    while time.time() - start_time < max_time - 30:
        prev_points = points.copy()
        
        # Compute gradient
        grad = _surrogate_gradient(points, beta=50.0)
        if grad is None:
            break
        
        # Update velocity with momentum
        velocity = momentum * velocity - step_size * grad
        
        # Take step
        points = _gradient_step(points, velocity, 1.0)
        
        # Calculate new covering radius
        current_rad = _covering_radius(points)
        
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = points.copy()
            fail_count = 0
        else:
            points = prev_points
            fail_count += 1
            if fail_count > 10:
                step_size *= 0.9
                velocity *= 0.5
                fail_count = 0
                if step_size < 0.0001:
                    break
    
    # Phase 2: Greedy circumcenter moves (original approach, refined)
    if time.time() - start_time < max_time - 20:
        points = best_points.copy()
        step_size = 0.02
        min_step_size = 0.0005
        fail_count = 0
        
        while time.time() - start_time < max_time - 20:
            prev_points = points.copy()
            
            try:
                hull = ConvexHull(points)
            except Exception:
                break
                
            # Find all facet radii and indices
            facet_radii = []
            facet_indices = []
            for idx, simplex in enumerate(hull.simplices):
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
                facet_radii.append(rad)
                facet_indices.append(simplex)
                
            if not facet_radii:
                break
                
            # Sort facets by radius (descending)
            sorted_indices = np.argsort(facet_radii)[::-1]
            
            # Move vertices of the top K worst facets
            K = 3
            moved_indices = set()
            for k in range(min(K, len(sorted_indices))):
                facet_idx = sorted_indices[k]
                simplex = facet_indices[facet_idx]
                
                # Recompute circumcenter for this specific facet
                v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                n = np.cross(v1 - v2, v1 - v3)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                c = n / norm_n
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                    
                # Move vertices towards circumcenter
                for j in range(3):
                    v_idx = simplex[j]
                    if v_idx in moved_indices:
                        continue
                    v = points[v_idx]
                    new_v = v + step_size * (c - v)
                    norm_v = np.linalg.norm(new_v)
                    if norm_v > 1e-10:
                        points[v_idx] = new_v / norm_v
                        moved_indices.add(v_idx)
                        
            # Calculate new covering radius
            current_rad = _covering_radius(points)
            
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
                fail_count = 0
            else:
                points = prev_points
                fail_count += 1
                if fail_count > 8:
                    step_size *= 0.85
                    fail_count = 0
                    if step_size < min_step_size:
                        break
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < max_time - 5:
        points = best_points.copy()
        step_size = 0.005
        for _ in range(30):
            if time.time() - start_time > max_time - 5:
                break
            prev_points = points.copy()
            try:
                hull = ConvexHull(points)
            except Exception:
                break
            max_facet_rad = 0.0
            worst_circumcenter = None
            worst_indices = None
            for idx, simplex in enumerate(hull.simplices):
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
            if worst_circumcenter is None:
                break
            for j in range(3):
                v_idx = worst_indices[j]
                v = points[v_idx]
                new_v = v + step_size * (worst_circumcenter - v)
                points[v_idx] = new_v / np.linalg.norm(new_v)
            current_rad = _covering_radius(points)
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
            else:
                points = prev_points
    
    return best_points
