import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _generate_test_points(n_test=5000):
    """Generate test points on sphere for covering radius estimation."""
    # Use a combination of Fibonacci spiral and random points
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _covering_radius_smooth(points, test_points, temperature=0.1):
    """Compute smooth approximation of covering radius using soft-max."""
    # Compute angular distances from each test point to each configuration point
    # dist[i,j] = arccos(dot(test_points[i], points[j]))
    dots = test_points @ points.T  # (n_test, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_test, N)
    
    # For each test point, find min distance to any configuration point
    min_dists = np.min(dists, axis=1)  # (n_test,)
    
    # Soft-max approximation of max(min_dists)
    # Use log-sum-exp for numerical stability
    scaled = min_dists / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = temperature * log_sum
    
    return soft_max

def _covering_radius_exact(points, test_points):
    """Compute exact covering radius (max of min distances)."""
    dots = test_points @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    return np.max(min_dists)

def _covering_radius_hull(points):
    """Compute covering radius using convex hull circumradii."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Invalid hull
    
    max_radius = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute circumcenter of triangle on unit sphere
        # Using the formula: circumcenter is proportional to (p1×p2 + p2×p3 + p3×p1)
        # or use the plane intersection method
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            continue
        normal = normal / normal_norm
        
        # The circumcenter lies along the normal direction
        # Check which side
        center = normal
        if np.dot(center, p1) < 0 or np.dot(center, p2) < 0 or np.dot(center, p3) < 0:
            center = -center
        
        dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        max_radius = max(max_radius, dist)
    
    return max_radius

def _optimize_smooth(points, deadline, temperature=0.1):
    """Optimize using smooth soft-max covering radius objective."""
    test_points = _generate_test_points(3000)
    
    def objective(x):
        pts = _normalize(x.reshape(N, 3))
        return _covering_radius_smooth(pts, test_points, temperature)
    
    x0 = points.flatten()
    
    # Use L-BFGS-B with multiple temperature annealing steps
    temps = [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03]
    current_pts = points.copy()
    
    for temp in temps:
        if time.time() > deadline - 10:
            break
        
        def obj_temp(x):
            pts = _normalize(x.reshape(N, 3))
            return _covering_radius_smooth(pts, test_points, temp)
        
        result = minimize(
            obj_temp, x0, method='L-BFGS-B',
            options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.fun < 1.0:  # Valid result
            current_pts = _normalize(result.x.reshape(N, 3))
            x0 = result.x
        else:
            break
    
    return current_pts

def _local_refinement(points, deadline):
    """Local refinement: move points toward their Voronoi cell centroids."""
    test_points = _generate_test_points(5000)
    current_pts = points.copy()
    best_pts = points.copy()
    best_radius = _covering_radius_exact(current_pts, test_points)
    
    step_size = 0.02
    no_improve = 0
    max_no_improve = 20
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Compute centroid of nearest test points for each configuration point
        dots = test_points @ current_pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_idx = np.argmin(dists, axis=1)
        
        # For each configuration point, average the test points assigned to it
        centroids = np.zeros_like(current_pts)
        counts = np.zeros(N)
        for i in range(N):
            mask = (min_idx == i)
            if np.sum(mask) > 0:
                centroids[i] = np.mean(test_points[mask], axis=0)
                counts[i] = np.sum(mask)
        
        # Normalize centroids
        valid = counts > 0
        centroids[valid] /= np.linalg.norm(centroids[valid], axis=1, keepdims=True)
        
        # Move points toward centroids
        new_pts = current_pts.copy()
        for i in range(N):
            if counts[i] == 0:
                continue
            direction = centroids[i] - current_pts[i]
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                continue
            direction = direction / dir_norm
            new_pts[i] = current_pts[i] + step_size * direction
            new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
        
        new_radius = _covering_radius_exact(new_pts, test_points)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            step_size *= 0.9
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    
    # Phase 1: Smooth optimization with temperature annealing
    try:
        optimized = _optimize_smooth(pts, deadline - 30.0)
        if np.all(np.isfinite(optimized)):
            pts = optimized
    except Exception:
        pass
    
    # Phase 2: Local refinement
    try:
        refined = _local_refinement(pts, deadline - 5.0)
        if np.all(np.isfinite(refined)):
            pts = refined
    except Exception:
        pass
    
    # Final validation
    pts = _normalize(pts)
    if not np.all(np.isfinite(pts)):
        pts = _fibonacci_spiral(N)
    
    return pts
