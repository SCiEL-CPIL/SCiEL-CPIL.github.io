"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _angular_distance(x, y):
    """Compute angular distance (in radians) between two unit vectors."""
    dot = np.clip(np.sum(x * y, axis=-1), -1.0, 1.0)
    return np.arccos(dot)


def _covering_radius(points):
    """Compute the covering radius (in radians) of a set of points on the unit sphere.
    
    The covering radius is the maximum over all points on the sphere of the angular
    distance to the nearest point in the configuration. This equals the largest
    circumradius of the facets of the convex hull.
    
    We approximate this by:
    1. Computing the convex hull of the points
    2. For each facet, finding its circumcenter (on the sphere) and the angular
       distance from that center to the facet's vertices
    """
    from scipy.spatial import ConvexHull
    
    points = np.asarray(points)
    if len(points) < 4:
        return np.pi / 2.0
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based estimate
        return _covering_radius_grid(points)
    
    max_r = 0.0
    for simplex in hull.simplices:
        # Get the vertices of this facet
        verts = points[simplex]
        # The circumcenter of a spherical triangle/polygon can be found
        # by finding the point on the sphere equidistant from all vertices
        # For a simplex (triangle or tetrahedron facet in 3D), we can compute
        # the circumcenter as the normalized cross product approach or solve
        # for the point equidistant from all vertices
        
        if len(verts) == 3:
            # Triangle: circumcenter is the normalized vector that is equidistant
            # from all three vertices
            a, b, c = verts
            # The circumcenter lies along the normal to the plane of the triangle,
            # but on the sphere. For a spherical triangle, the circumcenter is
            # the point on the sphere equidistant from a, b, c.
            # This can be found by solving: |x - a|^2 = |x - b|^2 = |x - c|^2, |x|=1
            # |x|^2 - 2x.a + |a|^2 = |x|^2 - 2x.b + |b|^2
            # -2x.a + 1 = -2x.b + 1 => x.a = x.b
            # Similarly x.a = x.c
            # So x is in the span of (a-b) and (a-c) orthogonal complement... 
            # Actually, x is along the direction of the cross product of two edges
            # from the circumcenter. Let's use a different approach.
            # The circumcenter of a spherical triangle is along the direction:
            # x = normalize(a * (b x c) + b * (c x a) + c * (a x b)) / |...|
            # Wait, that's not quite right either. Let me think more carefully.
            # For a point x on the sphere equidistant from a, b, c:
            # x . a = x . b = x . c
            # x is orthogonal to (a-b) and (a-c)
            # So x is parallel to (a-b) x (a-c)
            # But we need x on the sphere, so x = normalize((a-b) x (a-c))
            # Wait, that gives the normal to the plane, which is the circumcenter
            # only if the triangle is right... No.
            # Actually, x . a = x . b means x . (a - b) = 0
            # x . a = x . c means x . (a - c) = 0
            # So x is orthogonal to both (a-b) and (a-c), hence x is parallel to
            # (a-b) x (a-c). The circumcenter is the normalized version of this.
            # But wait, this is the circumcenter of the planar triangle, not the
            # spherical one. For the spherical case, since a, b, c are on the unit
            # sphere, the point x = normalize((a-b) x (a-c)) satisfies x . a = x . b = x . c
            # because x is orthogonal to (a-b) and (a-c).
            # The angular distance from x to a is arccos(x . a).
            n = np.cross(a - b, a - c)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            x = n / norm_n
            # Ensure x is on the correct side (same hemisphere as the triangle)
            # The circumcenter should be on the same side as the triangle's centroid
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            r = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
            max_r = max(max_r, r)
        else:
            # For non-triangular facets (shouldn't happen in 3D convex hull),
            # fall back to checking vertices
            for v in verts:
                # Just use the angular distance between pairs as a rough estimate
                pass
    
    return max_r


def _covering_radius_grid(points):
    """Fallback: estimate covering radius using a grid on the sphere."""
    points = np.asarray(points)
    # Create a grid of test points
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    X = np.sin(TH) * np.cos(PH)
    Y = np.sin(TH) * np.sin(PH)
    Z = np.cos(TH)
    grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    # Compute distances from each grid point to the nearest configuration point
    # Use chunked computation to avoid memory issues
    max_dist = 0.0
    chunk_size = 10000
    for i in range(0, len(grid), chunk_size):
        chunk = grid[i:i+chunk_size]
        # Compute dot products
        dots = np.clip(chunk @ points.T, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        dists = np.arccos(min_dots)
        max_dist = max(max_dist, np.max(dists))
    return max_dist


def _objective(points_flat):
    """Objective function: covering radius in radians."""
    points = points_flat.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    return _covering_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_score = _covering_radius(points)
    
    # Try a few optimization approaches
    # Use L-BFGS-B with a smooth surrogate: soft-max of squared distances
    
    def _smooth_objective(points_flat, beta=10.0):
        """Smooth surrogate for the covering radius using log-sum-exp."""
        points = points_flat.reshape(-1, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Use a moderate grid for the smooth objective
        n_theta = 50
        n_phi = 100
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        X = np.sin(TH) * np.cos(PH)
        Y = np.sin(TH) * np.sin(PH)
        Z = np.cos(TH)
        grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        
        # Compute distances
        dots = np.clip(grid @ points.T, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)
        dists = np.arccos(min_dots)
        
        # Log-sum-exp for smooth max
        max_d = np.max(dists)
        if max_d < 1e-10:
            return 0.0
        shifted = dists - max_d
        log_sum_exp = max_d + np.log(np.sum(np.exp(beta * shifted))) / beta
        return log_sum_exp
    
    # Gradient-free optimization using Nelder-Mead or similar
    # Since the objective is non-smooth, use a derivative-free method
    
    from scipy.optimize import minimize
    
    # Try several random restarts with local optimization
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current = best_points.copy()
        else:
            # Perturb the best solution
            current = best_points.copy()
            noise = np.random.normal(0, 0.1, size=current.shape)
            current = current + noise
            current = current / np.linalg.norm(current, axis=1, keepdims=True)
        
        # Use L-BFGS-B with the smooth objective
        try:
            result = minimize(
                _smooth_objective,
                current.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_score:
                new_points = result.x.reshape(-1, 3)
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                new_score = _covering_radius(new_points)
                if new_score < best_score:
                    best_score = new_score
                    best_points = new_points.copy()
        except Exception:
            pass
    
    # Final polish with smaller perturbations
    for polish_iter in range(10):
        if time.time() > deadline:
            break
        
        # Try small random perturbations
        improved = False
        for attempt in range(20):
            if time.time() > deadline:
                break
            perturbed = best_points.copy()
            mask = np.random.random(N) < 0.1  # Perturb ~10% of points
            if np.sum(mask) == 0:
                continue
            noise = np.random.normal(0, 0.01, size=perturbed.shape)
            perturbed[mask] += noise[mask]
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            score = _covering_radius(perturbed)
            if score < best_score - 1e-12:
                best_score = score
                best_points = perturbed.copy()
                improved = True
                break
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
