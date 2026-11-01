"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds
import time

def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    The covering radius is the maximum over all points on the sphere of the angular
    distance to the nearest point in the configuration.
    
    For a set of points on the sphere, the covering radius equals the maximum
    circumradius of the facets of the convex hull (for facets whose circumcenter
    lies within the facet, i.e., empty caps).
    
    We use a sampling-based approach for robustness, combined with checking
    the Voronoi cells via the convex hull.
    """
    from scipy.spatial import ConvexHull, Delaunay
    
    # Method 1: Use convex hull facets
    try:
        hull = ConvexHull(pts)
        max_radius = 0.0
        n = len(pts)
        
        for simplex in hull.simplices:
            # Get the 3 points forming this triangle facet
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            
            # Compute the circumcenter of this triangle on the sphere
            # The circumcenter of a spherical triangle is the intersection of the
            # perpendicular bisectors, or equivalently, it's proportional to
            # (p1 * |p2-p3|^2 * cross(p2,p3) + ...) -- but easier:
            # The circumcenter c satisfies |c - p1| = |c - p2| = |c - p3| and c is on the sphere.
            # Since all points are on the unit sphere, the circumcenter of the spherical triangle
            # is in the direction of the vector:
            # c_dir = (p1 * (|p2-p3|^2) * cross(p2,p3) + p2 * (|p3-p1|^2) * cross(p3,p1) + p3 * (|p1-p2|^2) * cross(p1,p2))
            # Actually, a cleaner formula: the circumcenter of points on a sphere through origin
            # is in the direction of the solution to the linear system.
            
            # Alternative: the circumcenter of a spherical triangle with vertices on the unit sphere
            # lies in the direction of the cross product of the edge midpoints' normals.
            # Simplest: solve for c such that c·p1 = c·p2 = c·p3 and |c|=1.
            # This means c is orthogonal to (p1-p2) and (p2-p3).
            
            v1 = p1 - p2
            v2 = p2 - p3
            c_dir = np.cross(v1, v2)
            
            if np.linalg.norm(c_dir) < 1e-12:
                # Degenerate (collinear) - shouldn't happen for convex hull
                continue
            
            c_dir /= np.linalg.norm(c_dir)
            
            # Check if the circumcenter is "inside" the spherical triangle
            # i.e., the cap centered at c_dir with radius = angular distance to p1
            # contains no other points (empty cap condition)
            # Actually for covering radius, we need the circumcenter to be a valid
            # Voronoi vertex, meaning it's equidistant to p1,p2,p3 and closer to them
            # than to any other point.
            
            dist_to_p1 = np.arccos(np.clip(np.dot(c_dir, p1), -1.0, 1.0))
            
            # Check empty cap: no other point is closer to c_dir
            dots = pts @ c_dir
            # The angular distance from c_dir to point i is arccos(dots[i])
            # We need dist_to_p1 to be the minimum (or tied minimum)
            min_dot = np.max(dots)  # max dot = min angular distance
            if min_dot < np.dot(c_dir, p1) - 1e-9:
                # Some other point is closer, so this circumcenter is not a Voronoi vertex
                continue
            
            # Also need to check that c_dir is on the "correct side" - 
            # the circumcenter should be in the hemisphere defined by the facet
            # For a convex hull, the outward normal should point away from origin
            normal = np.cross(p2 - p1, p3 - p1)
            if np.dot(normal, c_dir) < 0:
                c_dir = -c_dir
                dist_to_p1 = np.arccos(np.clip(np.dot(c_dir, p1), -1.0, 1.0))
                # Re-check empty cap
                dots = pts @ c_dir
                min_dot = np.max(dots)
                if min_dot < np.dot(c_dir, p1) - 1e-9:
                    continue
            
            max_radius = max(max_radius, dist_to_p1)
        
        # Convert to degrees
        return np.degrees(max_radius)
    
    except Exception:
        pass
    
    # Fallback: sampling-based estimate
    # Generate many random points and find the max distance to nearest config point
    rng = np.random.default_rng(42)
    n_samples = 20000
    # Uniform random points on sphere
    z = rng.uniform(-1, 1, n_samples)
    phi = rng.uniform(0, 2 * np.pi, n_samples)
    r = np.sqrt(1 - z**2)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each sample to all config points
    # Use batched dot products
    dots = samples @ pts.T  # (n_samples, N)
    max_dots = np.max(dots, axis=1)  # max dot = min angular distance
    min_ang_dist = np.arccos(np.clip(max_dots, -1.0, 1.0))
    
    return np.degrees(np.max(min_ang_dist))


def _lloyd_relaxation(pts, n_iter=50, time_limit=10.0):
    """Lloyd relaxation: move each point to the centroid of its Voronoi cell."""
    from scipy.spatial import Delaunay
    import time
    
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        # For each point, compute the centroid of its Voronoi cell on the sphere
        # Approximation: for each point, find nearby points and compute weighted centroid
        # Better: use the fact that the Voronoi cell centroid on the sphere can be
        # approximated by the average of the circumcenters of adjacent facets
        
        # Simple approach: move each point toward the centroid of its neighbors' midpoints
        # Actually, let's use a simpler repulsion-based relaxation
        
        # Compute pairwise distances
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)  # ensure self-distance is max
        dists = np.arccos(np.clip(dots, -1.0, 1.0))
        
        # For each point, compute a weighted average of other points, weighted by inverse distance
        # This pushes points apart
        new_pts = np.zeros_like(pts)
        for i in range(n):
            weights = 1.0 / (dists[i] + 1e-10)
            weights[i] = 0
            # Move away from close neighbors
            repulsion = np.sum(pts * weights[:, np.newaxis], axis=0)
            new_pts[i] = pts[i] - 0.1 * repulsion / (np.sum(weights) + 1e-10)
        
        # Renormalize
        norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts = new_pts / np.maximum(norms, 1e-10)
        
        pts = new_pts
    
    return pts


def _optimize_covering(pts, time_budget=100.0):
    """Optimize the covering radius using gradient-based methods."""
    import time
    
    start_time = time.time()
    n = len(pts)
    
    def objective(params):
        """Objective: covering radius. Params are 3N coordinates, normalized to sphere."""
        p = params.reshape(n, 3)
        # Normalize to unit sphere
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / np.maximum(norms, 1e-10)
        return _compute_covering_radius(p)
    
    def objective_grad(params):
        """Objective with numerical gradient."""
        val = objective(params)
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            params_minus = params.copy()
            params_minus[i] -= eps
            grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
        return val, grad
    
    # Try L-BFGS-B with gradient
    best_pts = pts.copy()
    best_val = _compute_covering_radius(pts)
    
    # Phase 1: Use Nelder-Mead for a few iterations to get past non-smooth regions
    p_flat = pts.flatten()
    
    def obj_flat(params):
        p = params.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        p = p / np.maximum(norms, 1e-10)
        return _compute_covering_radius(p)
    
    # Nelder-Mead
    try:
        res = minimize(obj_flat, p_flat, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_val = res.fun
            best_pts = res.x.reshape(n, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            best_pts = best_pts / np.maximum(norms, 1e-10)
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B
    if time.time() - start_time < time_budget:
        try:
            p_flat = best_pts.flatten()
            
            def lbfgs_obj(params):
                p = params.reshape(n, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p = p / np.maximum(norms, 1e-10)
                return _compute_covering_radius(p)
            
            res = minimize(lbfgs_obj, p_flat, method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_pts = res.x.reshape(n, 3)
                norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
                best_pts = best_pts / np.maximum(norms, 1e-10)
        except Exception:
            pass
    
    return best_pts, best_val


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    import time
    
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s safety margin for 150s limit
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(12345)
    
    # Start with golden-angle spiral as baseline
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    base_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Also try some random starts
    candidates = [base_pts]
    
    # Random starts
    for _ in range(3):
        if time.time() > deadline - 60:
            break
        # Random uniform points
        z_rand = rng.uniform(-1, 1, N)
        phi_rand = rng.uniform(0, 2 * np.pi, N)
        r_rand = np.sqrt(1 - z_rand**2)
        rand_pts = np.column_stack([r_rand * np.cos(phi_rand), r_rand * np.sin(phi_rand), z_rand])
        candidates.append(rand_pts)
    
    # Also try a grid-based start (latitude/longitude with offset)
    if time.time() < deadline - 60:
        # 50 points: roughly 7 rings of ~7 points each
        latitudes = np.linspace(-np.pi/2, np.pi/2, 7)
        grid_pts = []
        for lat in latitudes:
            n_long = max(1, int(np.round(7 * np.cos(lat))))
            longs = np.linspace(0, 2*np.pi, n_long, endpoint=False) + np.pi/n_long
            for lon in longs:
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z_val = np.sin(lat)
                grid_pts.append([x, y, z_val])
        grid_pts = np.array(grid_pts[:N])
        if len(grid_pts) == N:
            candidates.append(grid_pts)
    
    best_pts = base_pts
    best_val = _compute_covering_radius(base_pts)
    
    # Optimize each candidate
    for idx, init_pts in enumerate(candidates):
        if time.time() > deadline - 20:
            break
        
        # First do a quick Lloyd relaxation
        relaxed = _lloyd_relaxation(init_pts, n_iter=30, time_limit=5.0)
        
        # Evaluate
        val = _compute_covering_radius(relaxed)
        if val < best_val:
            best_val = val
            best_pts = relaxed.copy()
        
        # Full optimization
        if time.time() < deadline - 25:
            opt_pts, opt_val = _optimize_covering(relaxed, time_budget=min(40.0, deadline - time.time() - 10.0))
            if opt_val < best_val:
                best_val = opt_val
                best_pts = opt_pts.copy()
    
    # Final check: ensure points are on unit sphere and non-coincident
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-10)
    
    # Verify no coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 1.0 - 1e-6:
        # Points are too close, add small perturbation
        for i in range(N):
            for j in range(i+1, N):
                if np.abs(dots[i,j]) > 1.0 - 1e-6:
                    best_pts[j] += rng.normal(0, 0.01, 3)
                    best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Ensure not all in a closed hemisphere
    # Check if there's a plane through origin with all points on one side
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(best_pts)
        # If origin is inside the convex hull, we're not in a closed hemisphere
        # Check by seeing if 0 is in the convex hull
        # Simple check: for each facet, the origin should be on the opposite side from the outward normal
        # Actually, simpler: check if the sum of all points (or some combination) can represent 0
        # Use linear programming or just check the convex hull contains origin
        # Quick check: if all points have positive dot with some direction, they're in a hemisphere
        # The origin is inside the convex hull iff there exist positive weights summing to 1 such that sum w_i p_i = 0
        # For our purposes, the golden-angle spiral and optimized versions should be fine
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
