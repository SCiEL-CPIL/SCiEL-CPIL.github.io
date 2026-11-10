"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _spherical_circumcenters(points, facets):
    """
    Compute circumcenters of spherical triangles defined by points and facets.
    Returns an array of shape (n_facets, 3) with normalized circumcenter directions.
    """
    a = points[facets[:, 0]]
    b = points[facets[:, 1]]
    c = points[facets[:, 2]]
    
    # For unit vectors a,b,c, the circumcenter direction is:
    # n = (b-a) x (c-a) = bxc + axb + cxa
    # which equals a x b + b x c + c x a
    ab = np.cross(a, b)
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    
    n = ab + bc + ca
    norms_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    centers = np.zeros_like(a)
    valid = norms_n.squeeze() > 1e-12
    
    if np.any(valid):
        centers[valid] = n[valid] / norms_n[valid]
        
    return centers


def _covering_radius_fast(points):
    """
    Compute the covering radius in degrees using convex hull facets.
    The covering radius is the maximum circumradius of the spherical triangular facets.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        if len(facets) == 0:
            return 0.0
            
        centers = _spherical_circumcenters(points, facets)
        
        # For each facet, the circumradius is the angular distance from center to any vertex
        a = points[facets[:, 0]]
        
        cos_R = np.sum(a * centers, axis=1)
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        R = np.arccos(cos_R)
        return np.degrees(np.max(R))
    except Exception:
        return np.inf


def _lloyd_step(points):
    """
    Perform one iteration of Spherical Lloyd's Algorithm.
    Moves each point to the centroid of its Voronoi cell (approximated by avg of adjacent facet circumcenters).
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        centers = _spherical_circumcenters(points, facets)
        
        new_points = np.zeros_like(points)
        counts = np.zeros(len(points))
        
        # Accumulate circumcenters for each vertex
        np.add.at(new_points, facets[:, 0], centers)
        np.add.at(new_points, facets[:, 1], centers)
        np.add.at(new_points, facets[:, 2], centers)
        
        np.add.at(counts, facets[:, 0], 1)
        np.add.at(counts, facets[:, 1], 1)
        np.add.at(counts, facets[:, 2], 1)
        
        # Normalize
        valid = counts > 0
        new_points[valid] /= counts[valid, None]
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        valid_norms = norms.squeeze() > 1e-12
        new_points[valid_norms] /= norms[valid_norms]
        
        # Keep original points for invalid cases
        new_points[~valid] = points[~valid]
        new_points[~valid_norms] = points[~valid_norms]
        
        return new_points
    except Exception:
        return points


def _targeted_optimize(points, max_iters=200, step_size=0.05, rng=None, time_limit=None, start_time=None):
    """
    Targeted optimization: repeatedly identify the worst facet and perturb its vertices.
    This focuses computational effort on the region that determines the covering radius.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = points.copy()
    current_obj = _covering_radius_fast(current)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(current)
            facets = hull.simplices
            centers = _spherical_circumcenters(current, facets)
            
            # Compute circumradius for each facet
            a = current[facets[:, 0]]
            cos_R = np.sum(a * centers, axis=1)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            
            # Find the worst facet (largest circumradius)
            worst_idx = np.argmax(R)
            worst_facet = facets[worst_idx]
            
            # Perturb the vertices of the worst facet
            new_pts = current.copy()
            improved = False
            
            for v_idx in worst_facet:
                # Try several random perturbations
                for attempt in range(5):
                    direction = rng.standard_normal(3)
                    direction /= np.linalg.norm(direction)
                    
                    new_pts[v_idx] = current[v_idx] + step_size * direction
                    new_pts[v_idx] /= np.linalg.norm(new_pts[v_idx])
                    
                    new_obj = _covering_radius_fast(new_pts)
                    
                    if new_obj < current_obj:
                        current_obj = new_obj
                        current = new_pts.copy()
                        improved = True
                        break
                    
                    # Revert if not improved
                    new_pts[v_idx] = current[v_idx]
            
            if not improved:
                # Try a smaller step
                step_size *= 0.9
                if step_size < 0.001:
                    step_size = 0.05
                    # Try moving a random point instead
                    idx = rng.integers(0, len(current))
                    direction = rng.standard_normal(3)
                    direction /= np.linalg.norm(direction)
                    new_pts[idx] = current[idx] + 0.02 * direction
                    new_pts[idx] /= np.linalg.norm(new_pts[idx])
                    new_obj = _covering_radius_fast(new_pts)
                    if new_obj < current_obj:
                        current_obj = new_obj
                        current = new_pts
        except Exception:
            continue
    
    return current, current_obj


def _fibonacci_start(n, offset=0.5):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(n, rng):
    """Generate random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1)[:, None]
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    pts = _fibonacci_start(N)
    best_pts = pts.copy()
    best_obj = _covering_radius_fast(pts)
    
    # Phase 1: Lloyd's algorithm from multiple starts
    initializations = [
        _fibonacci_start(N, offset=0.5),
        _fibonacci_start(N, offset=0.3),
        _fibonacci_start(N, offset=0.7),
        _random_start(N, rng),
        _random_start(N, rng),
    ]
    
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        
        current = init.copy()
        
        # Run Lloyd's iterations
        for _ in range(30):
            if time.time() > deadline - 25.0:
                break
            current = _lloyd_step(current)
        
        r = _covering_radius_fast(current)
        if r < best_obj:
            best_obj = r
            best_pts = current.copy()
    
    # Phase 2: Targeted optimization on the best solution
    if time.time() < deadline - 10.0:
        refined_pts, refined_obj = _targeted_optimize(
            best_pts, max_iters=100, step_size=0.05, rng=rng,
            time_limit=deadline - time.time() - 5.0, start_time=start_time
        )
        if refined_obj < best_obj:
            best_obj = refined_obj
            best_pts = refined_pts
    
    # Phase 3: Another round of Lloyd's on the refined solution
    if time.time() < deadline - 5.0:
        current = best_pts.copy()
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            current = _lloyd_step(current)
        r = _covering_radius_fast(current)
        if r < best_obj:
            best_obj = r
            best_pts = current.copy()
    
    # Ensure all points are normalized
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1)[:, None]
    
    return best_pts
# EVOLVE-BLOCK-END
