"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(N, rng):
    """Generate N random points on the unit sphere."""
    pts = rng.randn(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _make_sample_grid(n=1000):
    """
    A fixed, dense, well-spread set of probe points on the unit sphere.
    We use a higher resolution grid for better accuracy of the fast estimator.
    """
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _cover_radius_fast(pts, sample):
    """
    Fast vectorized estimate of the covering radius (radians).
    Max over probe points of the angular distance to the nearest configuration
    point. Cheap (no ConvexHull) so it can be called many times per second.
    """
    # Nearest dot product (max) for each probe point and for each config point.
    # d_sample is (M, N), max over axis 1 gives (M,)
    d_sample = sample @ pts.T          
    max_dot_sample = d_sample.max(axis=1)
    
    # Also check the configuration points themselves, as the worst case might be 
    # near a point if the grid is sparse, though for covering radius, the worst 
    # point is usually in the 'holes' between points. 
    # However, the covering radius is defined as max_y min_p dist(y,p).
    # The maximum of min_dist is attained at a vertex of the Voronoi diagram 
    # (circumcenter of a triangle) or on the boundary. 
    # Sampling is an approximation. To be safe, we also check the points themselves 
    # (dist 0) which doesn't help the max, but the grid should catch the holes.
    
    # The covering radius is determined by the point y that is FURTHEST from 
    # its nearest neighbor. 
    # min_dist(y) = arccos(max_p dot(y,p)).
    # We want max_y min_dist(y).
    # This is equivalent to min_y max_p dot(y,p) being as small as possible? No.
    # We want to minimize R such that for all y, min_p dist(y,p) <= R.
    # This means for all y, max_p dot(y,p) >= cos(R).
    # So we want to maximize the minimum of (max_p dot(y,p)) over y.
    # Let f(y) = max_p dot(y,p). We want max_y f(y) to be as large as possible.
    # The value is cos(R*). So R* = arccos(max_y f(y)).
    
    min_max_dot = np.min(max_dot_sample)
    # Clip for numerical safety
    min_max_dot = np.clip(min_max_dot, -1.0, 1.0)
    return float(np.arccos(min_max_dot))


def _is_valid(pts):
    """Check if points are on sphere, distinct enough, and not in a hemisphere."""
    if not np.all(np.isfinite(pts)):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-6):
        return False
    
    # Check for coincident points
    # Min distance between any pair
    # O(N^2) is fine for N=50
    dists = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-4:
        return False
        
    # Check if in a hemisphere
    # Origin must be inside the convex hull
    try:
        hull = ConvexHull(pts)
        # If the hull is valid, we check if origin is inside.
        # A simpler check: if the points span the sphere, the origin is inside.
        # We can check if there is a vector u such that u·p > 0 for all p.
        # If such u exists, they are in a hemisphere.
        # Equivalent: min_p u·p < 0 for all u? No.
        # If they are in a hemisphere, there exists u such that u·p >= 0 for all p.
        # This is equivalent to the origin NOT being in the interior of the convex hull.
        
        # We can use the fact that if origin is inside, then for any direction, 
        # there are points on both sides.
        # Random check:
        rng = np.random.RandomState(0)
        for _ in range(10):
            u = rng.randn(3)
            u /= np.linalg.norm(u)
            dots = pts @ u
            if np.all(dots > 1e-6) or np.all(dots < -1e-6):
                return False
        return True
    except Exception:
        return False


def _simulated_annealing(pts, sample, time_budget=100.0):
    """
    Simulated Annealing to minimize covering radius.
    """
    pts = pts.copy()
    current_radius = _cover_radius_fast(pts, sample)
    best_pts = pts.copy()
    best_radius = current_radius
    
    start_time = time.time()
    T = 0.01  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < time_budget:
        # Check time more frequently
        if time.time() - start_time > time_budget:
            break
            
        # Pick a random point to move
        idx = np.random.randint(N)
        old_pt = pts[idx].copy()
        
        # Generate a new position: random direction perturbation
        # Move along the sphere surface
        # Pick a random vector, project it to the tangent space at old_pt
        v = np.random.randn(3)
        # Tangent space projection: v - (v·n)n where n=old_pt
        v = v - np.dot(v, old_pt) * old_pt
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            continue
        v = v / norm_v
        
        # Step size
        step_size = 0.05 * np.sqrt(T)
        new_pt = old_pt + step_size * v
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        pts[idx] = new_pt
        
        new_radius = _cover_radius_fast(pts, sample)
        
        delta = new_radius - current_radius
        
        if delta < 0:
            # Improvement
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = pts.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta / T):
                current_radius = new_radius
            else:
                # Reject
                pts[idx] = old_pt
                
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6

    return best_pts, best_radius


def _local_polish(pts, sample, time_budget=10.0):
    """
    Final local optimization using gradient-like steps on the worst points.
    """
    pts = pts.copy()
    current_radius = _cover_radius_fast(pts, sample)
    best_pts = pts.copy()
    best_radius = current_radius
    
    start_time = time.time()
    step = 0.01
    
    while time.time() - start_time < time_budget:
        if time.time() - start_time > time_budget:
            break
        
        # Find the worst sample point
        d_sample = sample @ pts.T
        max_dot_sample = d_sample.max(axis=1)
        worst_idx_sample = np.argmin(max_dot_sample)
        worst_y = sample[worst_idx_sample]
        
        # Find the nearest point to worst_y
        nearest_idx = np.argmax(max_dot_sample[worst_idx_sample]) # Wait, max_dot_sample is per sample point.
        # We need the nearest config point to worst_y.
        dots_to_y = pts @ worst_y
        nearest_idx = np.argmax(dots_to_y)
        
        # Move nearest_idx away from worst_y? No, towards worst_y to reduce distance.
        # But we must stay on sphere.
        # Gradient direction: from nearest_pt towards worst_y.
        dir_vec = worst_y - pts[nearest_idx]
        norm_d = np.linalg.norm(dir_vec)
        if norm_d < 1e-10:
            break
        dir_vec = dir_vec / norm_d
        
        # Project dir_vec to tangent space of pts[nearest_idx]
        n = pts[nearest_idx]
        dir_vec = dir_vec - np.dot(dir_vec, n) * n
        norm_dir = np.linalg.norm(dir_vec)
        if norm_dir < 1e-10:
            break
        dir_vec = dir_vec / norm_dir
        
        # Try moving
        new_pt = pts[nearest_idx] + step * dir_vec
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        pts[nearest_idx] = new_pt
        new_radius = _cover_radius_fast(pts, sample)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = pts.copy()
            current_radius = new_radius
            # Step up slightly?
            # step = min(step * 1.1, 0.05)
        else:
            # Revert
            pts[nearest_idx] = n
            step *= 0.9
            if step < 1e-5:
                break

    return best_pts, best_radius


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before the 150s wall-clock kill.

    sample = _make_sample_grid(1000)
    
    rng = np.random.RandomState(42)
    
    # Generate multiple starting configurations
    starts = []
    starts.append(_fibonacci_spiral(N))
    for _ in range(5):
        starts.append(_random_sphere_points(N, rng))
    
    best_pts = starts[0]
    best_radius = _cover_radius_fast(best_pts, sample)
    
    # Optimize each start with SA
    for s in starts:
        if time.time() > deadline - 20:
            break
        remaining = deadline - 20 - (time.time() - start_time)
        if remaining < 10:
            break
        budget = min(remaining * 0.5, 15.0)
        if budget < 5:
            continue
            
        opt, opt_rad = _simulated_annealing(s, sample, time_budget=budget)
        if opt_rad < best_radius:
            best_radius = opt_rad
            best_pts = opt
            
    # Final local polish on the best configuration
    if time.time() < deadline - 5:
        budget = deadline - 5 - (time.time() - start_time)
        if budget > 2:
            opt, opt_rad = _local_polish(best_pts, sample, time_budget=budget)
            if opt_rad < best_radius:
                best_radius = opt_rad
                best_pts = opt

    # Final normalization and validity check
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    if not _is_valid(best_pts):
        # Fallback to Fibonacci if invalid
        best_pts = _fibonacci_spiral(N)
        best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
        
    return best_pts
# EVOLVE-BLOCK-END
