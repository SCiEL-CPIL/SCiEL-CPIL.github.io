"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

def _compute_covering_radius(points):
    """Compute the covering radius of a set of points on the unit sphere.
    Returns the max angular distance from any point on the sphere to the nearest
    configuration point, in degrees."""
    # Build the convex hull of the points
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use Delaunay
        try:
            hull = Delaunay(points)
            # For Delaunay, the facets are in hull.simplices
            simplices = hull.simplices
        except Exception:
            return 90.0  # worst case
    
    if hasattr(hull, 'simplices'):
        simplices = hull.simplices
    else:
        simplices = hull.simplices
    
    max_radius = 0.0
    for simplex in simplices:
        # Get the points forming this facet
        facet_pts = points[simplex]
        
        # Compute the circumcenter of the facet (for a spherical cap, this is
        # the point on the sphere farthest from all facet vertices)
        # The circumcenter of a spherical triangle is the intersection of the
        # perpendicular bisectors of its edges, projected onto the sphere.
        
        # For a facet with vertices v1, v2, v3, the circumcenter is the
        # normalized vector that is equidistant from all three vertices.
        # We solve for a point c such that c·v1 = c·v2 = c·v3, |c| = 1.
        
        if len(simplex) == 3:
            v1, v2, v3 = facet_pts
            # The circumcenter is proportional to v1 x v2 + v2 x v3 + v3 x v1
            # (for a spherical triangle)
            circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
            norm = np.linalg.norm(circum)
            if norm < 1e-15:
                continue
            circum = circum / norm
            
            # Check that this is on the correct side (inside the cap)
            # The circumcenter should be farther from the origin's projection
            # than the vertices... actually, let's just compute the angular distance
            # from the circumcenter to each vertex and take the max
            cos_dists = np.clip(np.abs(np.dot(circum, facet_pts)), -1.0, 1.0)
            ang_dists = np.arccos(cos_dists)
            facet_radius = np.max(ang_dists)
            
        elif len(simplex) == 2:
            # Edge case: midpoint
            midpoint = (facet_pts[0] + facet_pts[1])
            norm = np.linalg.norm(midpoint)
            if norm < 1e-15:
                continue
            midpoint = midpoint / norm
            cos_dists = np.clip(np.abs(np.dot(midpoint, facet_pts)), -1.0, 1.0)
            ang_dists = np.arccos(cos_dists)
            facet_radius = np.max(ang_dists)
        else:
            continue
        
        max_radius = max(max_radius, facet_radius)
    
    return np.degrees(max_radius)


def _lloyd_relaxation(points, n_iter=100, tol=1e-10):
    """Lloyd relaxation: move each point toward the circumcenter of its Voronoi cell."""
    points = np.array(points, dtype=float)
    n = len(points)
    
    for iteration in range(n_iter):
        if iteration % 50 == 0 and iteration > 0:
            # Check convergence
            pass
        
        # Compute Voronoi cells using Delaunay triangulation
        try:
            delaunay = Delaunay(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        
        for i in range(n):
            # Find the Voronoi cell of point i
            # The Voronoi cell is the set of points closer to i than to any other
            # We can approximate by finding the circumcenters of all Delaunay
            # triangles that contain vertex i
            
            # Find all simplices containing vertex i
            cell_circumcenters = []
            for simplex in delaunay.simplices:
                if i in simplex:
                    facet_pts = points[simplex]
                    if len(facet_pts) == 3:
                        v1, v2, v3 = facet_pts
                        circum = np.cross(v1, v2) + np.cross(v2, v3) + np.cross(v3, v1)
                        norm = np.linalg.norm(circum)
                        if norm > 1e-15:
                            circum = circum / norm
                            cell_circumcenters.append(circum)
                    elif len(facet_pts) == 2:
                        midpoint = (facet_pts[0] + facet_pts[1])
                        norm = np.linalg.norm(midpoint)
                        if norm > 1e-15:
                            midpoint = midpoint / norm
                            cell_circumcenters.append(midpoint)
            
            if len(cell_circumcenters) > 0:
                # Average the circumcenters (on the sphere, use vector average)
                avg = np.mean(cell_circumcenters, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-15:
                    new_points[i] = avg / norm
                else:
                    new_points[i] = points[i]
            else:
                new_points[i] = points[i]
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / np.maximum(norms, 1e-15)
        
        # Check convergence
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        
        if diff < tol:
            break
    
    return points


def _soft_max_covering_radius(points, beta=500.0):
    """Compute a smooth surrogate for the covering radius using soft-max."""
    # Sample points on the sphere
    n_samples = 2000
    # Use Fibonacci spiral for good coverage
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distances from each sample to each point
    # cos(theta) = dot(sample, point)
    cos_dists = samples @ points.T  # shape (n_samples, n)
    cos_dists = np.clip(cos_dists, -1.0, 1.0)
    
    # For each sample, find the minimum angular distance (i.e., max cos_dist)
    max_cos = np.max(cos_dists, axis=1)
    
    # Soft-max over the worst samples
    # We want to maximize the minimum distance, so we minimize the max distance
    # Use log-sum-exp for numerical stability
    logit = np.log(1.0 - max_cos + 1e-15)  # approximates arccos for small angles
    # Actually, let's just use the angles directly
    angles = np.arccos(max_cos)
    
    # Soft-max: log(sum(exp(beta * angles))) / beta
    max_angle = np.max(angles)
    shifted = angles - max_angle
    log_sum_exp = np.log(np.sum(np.exp(beta * shifted))) + beta * max_angle
    soft_max = log_sum_exp / beta
    
    return soft_max


def _optimize_with_lloyd(points, time_budget=100.0):
    """Optimize points using Lloyd relaxation with time budget."""
    start_time = time.time()
    points = np.array(points, dtype=float)
    
    # First, do a few Lloyd iterations
    points = _lloyd_relaxation(points, n_iter=50, tol=1e-8)
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.5:
        return points
    
    # Now do gradient-based optimization on the soft-max surrogate
    def objective(x):
        pts = x.reshape(-1, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / np.maximum(norms, 1e-15)
        return _soft_max_covering_radius(pts, beta=500.0)
    
    x0 = points.flatten()
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.success:
            new_points = result.x.reshape(-1, 3)
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / np.maximum(norms, 1e-15)
            points = new_points
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_budget * 0.8:
        return points
    
    # Final Lloyd polish
    points = _lloyd_relaxation(points, n_iter=30, tol=1e-10)
    
    return points


def _generate_initial_configurations(n_configs=10):
    """Generate multiple initial configurations for diversity."""
    configs = []
    
    for cfg_idx in range(n_configs):
        if cfg_idx == 0:
            # Golden-angle spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        elif cfg_idx == 1:
            # Random points
            rng = np.random.RandomState(42 + cfg_idx)
            points = rng.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        elif cfg_idx == 2:
            # Perturbed golden-angle
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            rng = np.random.RandomState(100 + cfg_idx)
            noise = rng.randn(N, 3) * 0.1
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        else:
            # More random starts
            rng = np.random.RandomState(200 + cfg_idx * 7)
            points = rng.randn(N, 3)
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        configs.append(points)
    
    return configs


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate initial configurations
    configs = _generate_initial_configurations(n_configs=8)
    
    best_points = None
    best_radius = np.inf
    
    for cfg_idx, init_points in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            break
        
        # Optimize this configuration
        try:
            optimized = _optimize_with_lloyd(init_points, time_budget=10.0)
            radius = _compute_covering_radius(optimized)
            
            if radius < best_radius:
                best_radius = radius
                best_points = optimized
        except Exception:
            continue
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0 or best_points is None:
        # Fallback: return the best we have or a default
        if best_points is not None:
            return best_points
        else:
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Final polish with remaining time
    try:
        final_points = _lloyd_relaxation(best_points, n_iter=50, tol=1e-12)
        final_radius = _compute_covering_radius(final_points)
        if final_radius < best_radius:
            best_points = final_points
    except Exception:
        pass
    
    return best_points
# EVOLVE-BLOCK-END
