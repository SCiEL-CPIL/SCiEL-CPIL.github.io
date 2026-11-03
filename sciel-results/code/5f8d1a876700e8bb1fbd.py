import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians for a set of unit points.
    Uses convex hull facet circumradii.
    """
    if len(points) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi
        
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normals = normals / norms
    
    # Ensure outward-facing normals
    dots = np.sum(normals * p1, axis=1)
    normals = np.where(dots[:, None] < 0, -normals, normals)
    
    # The circumradius is the angle between the normal and any vertex of the facet
    h = np.sum(normals * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return np.max(angles)

def _spherical_lloyd(points, max_iter=200, damping=0.5, time_limit=30):
    """
    Spherical Lloyd's algorithm to minimize covering radius.
    Moves points towards the centroid of their adjacent facet normals.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        simplices = hull.simplices
        if len(simplices) == 0:
            break
            
        # Compute facet normals
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        normals = np.cross(v1, v2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        normals = normals / norms
        
        # Ensure outward-facing
        dots = np.sum(normals * p1, axis=1)
        normals = np.where(dots[:, None] < 0, -normals, normals)
        
        # Accumulate normals for each point
        # Each point is in multiple facets
        point_centroids = np.zeros((N, 3))
        
        for idx, simplex in enumerate(simplices):
            n = normals[idx]
            point_centroids[simplex[0]] += n
            point_centroids[simplex[1]] += n
            point_centroids[simplex[2]] += n
            
        # Normalize centroids to get target directions
        centroid_norms = np.linalg.norm(point_centroids, axis=1, keepdims=True)
        centroid_norms = np.where(centroid_norms < 1e-10, 1.0, centroid_norms)
        target_directions = point_centroids / centroid_norms
        
        # Update points
        # Move towards the centroid direction
        # We can use a simple step: new_p = normalize(p + damping * (target - p))
        # Or project the movement onto the tangent plane
        # Simple approach:
        new_points = points + damping * (target_directions - points)
        new_points = _normalize(new_points)
        
        new_cost = _covering_radius(new_points)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = new_points.copy()
            points = new_points
        else:
            # If cost increased, revert and reduce damping slightly or break
            # For simplicity, just revert and continue with same damping
            # Or break if we are stuck
            pass
            
    return best_points, best_cost

def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(seed):
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)

def _icosahedron_based_points():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = list(verts)
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.5
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        # Check minimum distance to existing points to avoid clustering
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.2:
            points.append(new_pt)
            
    return np.array(points[:N])

def _polish(points, time_limit=10):
    """
    Fine-tune the configuration using Nelder-Mead.
    """
    start_time = time.time()
    best_params = points.flatten()
    best_score = _covering_radius(points)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        # Check for coincident points
        # This is a bit expensive, but N=50 is small
        # We can skip this check in the objective for speed, 
        # as Nelder-Mead is unlikely to create exact coincidences 
        # and the hull will fail gracefully or return a large value.
        try:
            return _covering_radius(pts)
        except Exception:
            return np.pi

    try:
        res = minimize(objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except TimeoutError:
        pass
    except Exception:
        pass
        
    final_points = _normalize(best_params.reshape(N, 3))
    return final_points, best_score

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin before 150s kill
    
    initializations = [
        _fibonacci_points(),
        _random_points(42),
        _random_points(123),
        _icosahedron_based_points()
    ]
    
    best_points = None
    best_score = np.pi
    
    for init in initializations:
        if time.time() > deadline - 20: # Leave 20s for polishing and final checks
            break
            
        # Run Lloyd's algorithm
        lloyd_points, lloyd_score = _spherical_lloyd(init, max_iter=100, damping=0.5, time_limit=10)
        
        if lloyd_score < best_score:
            best_score = lloyd_score
            best_points = lloyd_points
            
    # Polish the best result found
    if best_points is not None and time.time() < deadline - 5:
        polished_points, polished_score = _polish(best_points, time_limit=10)
        if polished_score < best_score:
            best_score = polished_score
            best_points = polished_points
            
    # Final safety check
    if best_points is None:
        best_points = _fibonacci_points()
        
    return _normalize(best_points)
