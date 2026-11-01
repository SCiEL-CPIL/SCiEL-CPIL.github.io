import numpy as np
import scipy.spatial
import scipy.optimize
import time

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    """Pre-compute all (i,j,k) with i<j<k"""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _triangle_areas(points):
    """Compute all triangle areas (unnormalized, 0.5*|cross|)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    return _triangle_areas(points).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def calc_ratio(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    try:
        if len(points) < 3:
            return 0.0
        hull_area = _hull_area(points)
        if hull_area < 1e-10:
            return 0.0
        min_area = _min_triangle_area(points)
        return min_area / hull_area
    except Exception:
        return 0.0

def normalize_points(points):
    """
    Scale points so that the convex hull area is 1.0.
    """
    try:
        area = _hull_area(points)
        if area < 1e-10:
            return points
        scale_factor = 1.0 / np.sqrt(area)
        return points * scale_factor
    except Exception:
        return points

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon points."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _concentric_circles(n, num_rings=3):
    """Generate points on concentric circles."""
    points = []
    points_per_ring = [n // num_rings] * (num_rings - 1) + [n - (n // num_rings) * (num_rings - 1)]
    radii = np.linspace(0.2, 1.0, num_rings)
    
    for ring_idx, (num_pts, radius) in enumerate(zip(points_per_ring, radii)):
        if num_pts == 1:
            points.append([0.0, 0.0] if ring_idx == 0 else [radius, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
            ring_points = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
            points.extend(ring_points.tolist())
    
    return np.array(points[:n])

def _random_in_disk(n, seed=0):
    """Generate random points in unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 1, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _perturbed_polygon(n, radius=1.0, noise_level=0.1, seed=0):
    """Generate perturbed regular polygon."""
    base = _regular_polygon(n, radius)
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, noise_level, (n, 2))
    return base + noise

def _coordinate_descent_polish(points, max_iter=500, tol=1e-12):
    """
    Coordinate descent that targets the minimum-area triangle.
    For each iteration, find the bottleneck triangle and try to move
    one of its vertices to increase the minimum area.
    """
    points = points.copy()
    n = len(points)
    best_score = calc_ratio(points)
    
    for iteration in range(max_iter):
        areas = _triangle_areas(points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        i, j, k = _TRI_INDICES[min_idx]
        pi, pj, pk = points[i], points[j], points[k]
        
        # Try moving each vertex of the bottleneck triangle
        best_improvement = 0.0
        best_new_points = None
        
        for pt_idx in [i, j, k]:
            # Try several step sizes and directions
            for step in [0.01, 0.005, 0.002, 0.001]:
                # Try 8 directions around the current point
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    new_points = points.copy()
                    new_points[pt_idx] = points[pt_idx] + step * direction
                    
                    new_score = calc_ratio(new_points)
                    if new_score > best_score + best_improvement:
                        best_improvement = new_score - best_score
                        best_new_points = new_points
        
        if best_new_points is not None:
            points = best_new_points
            best_score += best_improvement
            if best_improvement < tol:
                break
        else:
            # No improvement found with current step sizes; try smaller steps
            break
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Regular polygon
    initial_configs.append(_regular_polygon(N, 1.0))
    
    # Concentric circles with different parameters
    for num_rings in [2, 3, 4]:
        initial_configs.append(_concentric_circles(N, num_rings))
    
    # Perturbed polygons with different noise levels
    for noise in [0.05, 0.1, 0.15, 0.2]:
        for seed in [0, 1, 2]:
            initial_configs.append(_perturbed_polygon(N, 1.0, noise, seed))
    
    # Random configurations
    for seed in [0, 1, 2, 3, 4]:
        initial_configs.append(_random_in_disk(N, seed))
    
    # Process each initial configuration
    for idx, init_points in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        # Normalize the initial configuration
        points = normalize_points(init_points.copy())
        score = calc_ratio(points)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Apply coordinate descent polish
        if time.time() < deadline - 10:
            polished = _coordinate_descent_polish(points, max_iter=300)
            polished = normalize_points(polished)
            polished_score = calc_ratio(polished)
            
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
        
        # If we have time, try L-BFGS-B from the polished point
        if time.time() < deadline - 5 and best_points is not None:
            try:
                def lbfgs_obj(x):
                    pts = x.reshape(-1, 2)
                    return -calc_ratio(pts)
                
                res = scipy.optimize.minimize(
                    lbfgs_obj,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                opt_points = res.x.reshape(-1, 2)
                opt_points = normalize_points(opt_points)
                opt_score = calc_ratio(opt_points)
                
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass
    
    # Final return
    if best_points is None:
        best_points = normalize_points(_regular_polygon(N, 1.0))
    else:
        best_points = normalize_points(best_points)
        
    return best_points
