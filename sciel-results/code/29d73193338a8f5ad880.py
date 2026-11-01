"""
Covering the sphere with 130 equal caps.

Strategy:
1. Start from a Fibonacci spiral (good initial distribution).
2. Use a repulsion-based optimization (Ripley-type) to minimize covering radius.
3. The objective is the maximum angular distance to the nearest point (covering radius).
4. We use a smooth approximation via log-sum-exp to make it differentiable.
5. Use scipy.optimize.minimize with L-BFGS-B for optimization.
6. Final polish with a more precise local optimization.

The key insight: the covering radius is determined by the largest empty cap.
We can approximate this by maximizing the minimum distance between points,
but more directly, we can minimize the maximum distance from any sphere point
to the nearest configuration point.

A practical approach: minimize the sum of exponential distances (soft-max of
pairwise distances) which correlates with the covering radius, then refine.

Actually, a better approach for covering:
- The covering radius is the largest circumradius of the convex hull facets.
- Equivalently, it's the maximum over all points y on the sphere of min_i angle(y, p_i).
- We can approximate this by placing points to maximize the minimum pairwise distance
  (Tammes problem) which gives good covering, but not optimal.
- Better: directly minimize a smooth surrogate of the covering radius.

For the smooth surrogate:
  f(p) = log(sum_j exp(beta * d_j)) / beta
where d_j is the angular distance from a grid point j to its nearest p_i.
As beta increases, this approaches the max.

But computing this requires evaluating distances from many grid points to all p_i.

Alternative: Use the fact that the covering radius equals the maximum over the
Voronoi vertices (on the sphere) of the distance to the nearest point. The Voronoi
vertices are the circumcenters of the Delaunay triangulation facets.

Practical algorithm:
1. Initialize with Fibonacci spiral.
2. Iteratively:
   a. Compute the covering radius by checking all pairs and using the
      circumcenters of the Delaunay triangulation.
   b. Use gradient descent / L-BFGS to minimize a smooth approximation.
3. Polish with high-precision local search.

Given the 150s time limit, we need to be efficient. Let's use:
- A moderate number of iterations of a repulsion-based method.
- The repulsion force is based on the gradient of the soft-max covering radius.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay

N = 130

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 130  # seconds, leave 20s margin for the evaluator's 150s limit


def _elapsed():
    return time.time() - _START_TIME


def _time_exhausted(margin=5.0):
    return _elapsed() > _TIME_LIMIT - margin


def _normalize(v):
    """Normalize vectors to unit length, handling zero vectors."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _covering_radius(points):
    """
    Compute the covering radius (in radians) of a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This is equivalent to the largest circumradius
    of the convex hull facets.
    
    We compute this by finding the Delaunay triangulation on the sphere and
    computing the circumradius of each facet (triangle).
    """
    points = _normalize(points)
    
    # We need to compute the convex hull in 3D
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use pairwise distances
        return _covering_radius_fallback(points)
    
    # For each facet (triangle) of the convex hull, compute the circumradius
    # The circumradius of a triangle on the unit sphere is the angle from the
    # sphere center to the circumcenter of the triangle.
    
    # Actually, for a triangle with vertices a, b, c on the unit sphere,
    # the circumradius (angular) is the angle subtended at the center by the
    # circumcircle. The circumcenter in 3D is the point equidistant from a, b, c.
    # For points on the unit sphere, the circumcenter of the triangle (as a 3D point)
    # lies along the direction of a+b+c (for an acute triangle) or needs more care.
    
    # Simpler approach: the covering radius is the max over all points y on the sphere
    # of min_i angle(y, p_i). The maximum occurs at a Voronoi vertex, which is the
    # circumcenter of a Delaunay triangle.
    
    # For a triangle with vertices a, b, c on the unit sphere:
    # The circumcenter (on the sphere) is the point y such that |y-a| = |y-b| = |y-c|
    # and y is on the unit sphere. This is the normalized vector (a+b+c) if the
    # triangle is acute, but we need to be more careful.
    
    # Actually, the circumcenter of the triangle in 3D (not necessarily on the sphere)
    # is the point equidistant from a, b, c. For points on the unit sphere, the
    # circumradius r_circ satisfies:
    #   r_circ^2 = (a-b)^2 (a-c)^2 (b-c)^2 / (4 * area^2)  [in 3D Euclidean]
    # But we want the angular covering radius, which is the angle from the origin
    # to the circumcenter... no, that's not right either.
    
    # Let me think again. The covering radius is the maximum over y on the sphere
    # of min_i angle(y, p_i). The worst point y is at the center of the largest
    # empty circle. On the sphere, this corresponds to the circumcenter of the
    # Delaunay triangle that contains the largest empty circle.
    
    # For a triangle with vertices a, b, c on the unit sphere, the circumcenter
    # of the triangle (as a point in R^3) is:
    #   c_circ = (a + b + c) / |a + b + c|  (if the triangle is acute)
    # But this is only the circumcenter if the triangle is acute. For obtuse
    # triangles, the circumcenter is outside the triangle.
    
    # The angular distance from the circumcenter to any vertex is the circumradius.
    # For a triangle with side lengths (angular) alpha, beta, gamma, the circumradius
    # R satisfies: R = a / (2 sin(A)) where a is a side and A is the opposite angle.
    
    # Simpler: compute the circumcenter in 3D, then the angle from origin to circumcenter
    # is NOT the covering radius. The covering radius is the angular distance from
    # the circumcenter (on the sphere) to the nearest vertex.
    
    # Let me just compute it directly:
    # For each facet, find the circumcenter on the sphere and compute the max distance
    # to the three vertices.
    
    facets = hull.simplices
    max_radius = 0.0
    
    for tri in facets:
        a = points[tri[0]]
        b = points[tri[1]]
        c = points[tri[2]]
        
        # Find the circumcenter of triangle abc on the unit sphere
        # The circumcenter y satisfies:
        #   y . a = y . b = y . c  (equal angular distance to all three)
        #   |y| = 1
        #
        # From y.a = y.b: y . (a - b) = 0
        # From y.a = y.c: y . (a - c) = 0
        # So y is perpendicular to (a-b) and (a-c), i.e., y is parallel to (a-b) x (a-c)
        
        cross = np.cross(a - b, a - c)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        y = cross / norm_cross
        
        # Check if y is on the correct side (same hemisphere as the triangle)
        # The circumcenter should be such that the triangle vertices are "around" it
        # For a convex hull facet, the circumcenter should be on the same side as the
        # normal of the facet.
        
        # Compute the angular distances from y to a, b, c
        d_a = np.arccos(np.clip(np.dot(y, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(y, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(y, c), -1, 1))
        
        # The covering radius contribution is the max of these three
        r = max(d_a, d_b, d_c)
        
        # But we also need to check if y is the correct circumcenter.
        # If the triangle is obtuse, the circumcenter is on the opposite side.
        # In that case, the covering radius is determined by the midpoint of the
        # longest side.
        
        # Check if the triangle is obtuse by checking if y is on the opposite side
        # of the triangle from the origin.
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            continue
        normal = normal / normal_norm
        
        # The origin should be on the opposite side of the facet from the circumcenter
        # for the circumcenter to be the valid one.
        # Actually, for a convex hull, the origin is inside, so the facet normal
        # points outward. The circumcenter should be on the same side as the normal.
        
        # Check: if dot(y, normal) < 0, then y is on the wrong side, and the
        # covering radius is determined by the longest side's midpoint.
        if np.dot(y, normal) < 0:
            # Obtuse triangle: the covering radius is half the longest side
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            r = max(d_ab, d_bc, d_ca) / 2.0
        else:
            r = max(d_a, d_b, d_c)
        
        if r > max_radius:
            max_radius = r
    
    return max_radius


def _covering_radius_fallback(points):
    """Fallback covering radius computation using a grid."""
    # Use a spherical grid to approximate the covering radius
    n_theta = 100
    n_phi = 200
    
    max_dist = 0.0
    for i in range(n_theta):
        theta = np.pi * (i + 0.5) / n_theta
        for j in range(n_phi):
            phi = 2 * np.pi * (j + 0.5) / n_phi
            y = np.array([np.sin(theta) * np.cos(phi),
                          np.sin(theta) * np.sin(phi),
                          np.cos(theta)])
            # Distance to nearest point
            dots = points @ y
            max_dot = np.max(dots)
            dist = np.arccos(np.clip(max_dot, -1, 1))
            if dist > max_dist:
                max_dist = dist
    
    return max_dist


def _soft_covering_radius(points, beta=50.0, n_grid=5000):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    
    We sample points on the sphere and compute the soft-max of the distances
    to the nearest configuration point.
    """
    # Generate random points on the sphere
    rng = np.random.default_rng(42)
    grid = rng.normal(size=(n_grid, 3))
    grid = _normalize(grid)
    
    # Compute distances from each grid point to its nearest configuration point
    # dots[i, j] = dot(grid[i], points[j])
    dots = grid @ points.T
    max_dots = np.max(dots, axis=1)
    min_dists = np.arccos(np.clip(max_dots, -1, 1))
    
    # Soft max: (1/beta) * log(sum(exp(beta * dist)))
    scaled = beta * min_dists
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_max = log_sum_exp / beta
    
    return soft_max


def _gradient_soft_covering_radius(points, beta=50.0, n_grid=5000):
    """
    Compute the gradient of the soft covering radius with respect to the points.
    
    This is complex. Instead, let's use a different approach:
    We'll use the repulsion method where each point is pushed away from others
    based on the local density.
    """
    # For now, return None to indicate we don't have a gradient
    return None


def _repulsion_step(points, step_size=0.01, n_iter=1):
    """
    Perform a repulsion-based step to improve the covering.
    
    Each point is moved in the direction that increases the minimum distance
    to other points, weighted by the local covering radius.
    """
    for _ in range(n_iter):
        if _time_exhausted():
            break
        
        n = len(points)
        # Compute pairwise dot products
        dots = points @ points.T
        np.fill_diagonal(dots, 1.0)
        
        # Angular distances
        dists = np.arccos(np.clip(dots, -1, 1))
        
        # For each point, compute the direction to move
        # We want to move each point away from its neighbors
        # The force on point i is the sum over j of (p_i - p_j) / dist_ij^2
        # (Coulomb-like repulsion)
        
        # Compute forces
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-6:
                    continue
                # Force direction: away from j
                force_dir = points[i] - points[j]
                force_norm = np.linalg.norm(force_dir)
                if force_norm < 1e-12:
                    continue
                force_dir = force_dir / force_norm
                # Force magnitude: inverse square of distance
                force_mag = 1.0 / (d * d + 1e-6)
                forces[i] += force_dir * force_mag
        
        # Normalize forces to be tangent to the sphere
        for i in range(n):
            # Project force onto tangent space
            radial = np.dot(forces[i], points[i]) * points[i]
            tangent = forces[i] - radial
            forces[i] = tangent
        
        # Update points
        points_new = points + step_size * forces
        points_new = _normalize(points_new)
        
        # Check if the covering radius improved
        old_cr = _covering_radius(points)
        new_cr = _covering_radius(points_new)
        
        if new_cr < old_cr:
            points = points_new
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return points, step_size


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    
    Returns an array of shape (130, 3) with unit vectors.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Repulsion-based optimization
    step_size = 0.05
    max_iter = 2000
    no_improve_count = 0
    patience = 100
    
    for iteration in range(max_iter):
        if _time_exhausted(margin=10.0):
            break
        
        points, step_size = _repulsion_step(points, step_size, n_iter=1)
        
        cr = _covering_radius(points)
        
        if cr < best_cr - 1e-10:
            best_cr = cr
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                # Try a smaller step size
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-8:
                    break
    
    # Final polish: use scipy.optimize.minimize on a subset of points
    if not _time_exhausted(margin=30.0):
        try:
            # Use the best points as initial guess
            x0 = best_points.flatten()
            
            def objective(x):
                p = x.reshape(N, 3)
                p = _normalize(p)
                return _covering_radius(p)
            
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            
            if result.success or result.fun < best_cr:
                new_points = _normalize(result.x.reshape(N, 3))
                new_cr = _covering_radius(new_points)
                if new_cr < best_cr:
                    best_points = new_points
                    best_cr = new_cr
        except Exception:
            pass
    
    # Ensure the result is valid
    best_points = _normalize(best_points)
    
    # Check for coincident points
    dots = best_points @ best_points.T
    np.fill_diagonal(dots, 0.0)
    if np.max(dots) > 0.9999:
        # Perturb coincident points
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i+1, N):
                if np.dot(best_points[i], best_points[j]) > 0.9999:
                    perturbation = rng.normal(size=3) * 0.01
                    best_points[i] = _normalize(best_points[i] + perturbation)[0]
    
    return best_points


# Run construct at module level to capture the result
# Actually, we should not run it at module level. The evaluator will call construct().
# But we need to make sure the function is properly defined.
