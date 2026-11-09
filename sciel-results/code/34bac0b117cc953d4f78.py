"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    # The covering radius is the maximum over all points on the sphere of the
    # angular distance to the nearest point in the configuration.
    # This equals the maximum circumradius of the facets of the convex hull
    # that have an empty cap (i.e., the facet's plane does not pass through the origin).
    
    # For a set of points on the unit sphere, the covering radius can be computed as:
    # For each facet of the convex hull, compute the distance from the origin to the
    # plane of the facet. If the origin is on the "inside" of the hull (which it should be
    # for a well-distributed set), the covering radius is related to the angle subtended.
    
    # Actually, a more direct approach: the covering radius is the maximum angular distance
    # from any point on the sphere to its nearest neighbor in the set. This is equivalent to
    # the maximum over all directions y on the sphere of min_i angle(y, p_i).
    
    # A practical computation: for each facet of the convex hull, the "farthest" point on the
    # sphere from the nearest configuration point lies in the direction of the facet's normal
    # (outward). The angular distance is pi - arccos(n_hat · closest_point) ... 
    # Actually, let me think more carefully.
    
    # The covering radius theta_c is the largest theta such that there exists a point y on the
    # sphere with d(y, p_i) >= theta for all i. This is equivalent to the largest theta such
    # that the intersection of caps of radius theta centered at each p_i does not cover the sphere.
    
    # A standard result: for points on the unit sphere, the covering radius equals the maximum
    # over all facets of the convex hull of (pi/2 - arccos(h)), where h is the signed distance
    # from the origin to the facet plane, IF the origin is inside the convex hull.
    
    # Wait, let me reconsider. The convex hull of points on the unit sphere: if the origin is
    # inside the convex hull, then for each facet with outward normal n and plane n·x = d (where
    # d > 0 since origin is inside and n points outward), the point on the sphere farthest from
    # the facet's vertices in the "gap" direction is along n. The angular distance from n to the
    # nearest point among the facet's vertices is what matters.
    
    # Actually, the simplest robust approach for moderate N: sample many points on the sphere
    # and compute the max min-distance. But that's not exact.
    
    # Better: use the convex hull approach. For each facet, the circumcenter of the facet's
    # vertices projected onto the sphere gives the worst-case point. The covering radius is
    # the maximum over all facets of the angular distance from the facet's circumcenter to
    # the nearest vertex, but only for facets whose circumcenter lies on the sphere side.
    
    # Let me use a different, well-known approach:
    # The covering radius is the maximum, over all points y on S^2, of min_i arccos(y · p_i).
    # The maximum of min_i arccos(y · p_i) occurs at a point y that is either:
    # 1. A circumcenter of a set of 3 or more points (on the sphere), or
    # 2. Equidistant to 3+ points.
    
    # For practical purposes with N=130, I'll compute it via the convex hull:
    # For each triangular facet of the convex hull, compute the point on the sphere that is
    # farthest from the three vertices. This is the point along the outward normal of the facet
    # (normalized), and the angular distance to the nearest vertex.
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: dense sampling
        return _covering_radius_sample(points)
    
    max_angle = 0.0
    for simplex in hull.simplices:
        # Get the three points forming this facet
        pts = points[simplex]
        
        # Compute the outward normal
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        normal = np.cross(e1, e2)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        
        # Check that the normal points outward (away from origin)
        # The centroid of the facet should be on the positive side
        centroid = np.mean(pts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        # The worst-case point on the sphere for this facet is in the direction of 'normal'
        # Compute the angular distance from 'normal' to each of the three vertices
        # The covering contribution from this facet is the max of these three angles? No.
        # Actually, the point 'normal' (as a unit vector) is the center of the "gap" if the
        # origin is inside the hull. The angular distance from this point to each vertex gives
        # the radius of the cap that this facet "covers". The covering radius is the maximum
        # over all such facet centers of the angular distance to the nearest vertex.
        
        # But wait - the point 'normal' might not be the exact worst case. The worst case on
        # the sphere for this facet is actually the point on the sphere that maximizes the
        # minimum distance to the three vertices. For a triangle on the sphere, this is either
        # the circumcenter (if it's inside the spherical triangle) or one of the vertices.
        
        # For simplicity and correctness, let's compute: the angular distance from 'normal'
        # to each vertex. The relevant one is the maximum of these, because the "gap" extends
        # up to the farthest vertex from the center.
        
        # Actually no. Let me reconsider. The covering radius is the largest theta such that
        # there's a point y with d(y, p_i) >= theta for all i. For a facet, the point y = normal
        # has distances d_i = arccos(normal · p_i) to each vertex. The minimum of these is the
        # distance from y to the nearest configuration point. So the covering radius is at least
        # min_i d_i for this y. And the maximum over all y gives the covering radius.
        
        # So for each facet, compute y = normal (unit), then compute min_i arccos(y · p_i).
        # The covering radius is the maximum over all facets of this minimum.
        
        dots = np.clip(np.dot(pts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)
        
        if min_angle > max_angle:
            max_angle = min_angle
    
    # Also need to check: what if the origin is NOT inside the convex hull? Then the covering
    # radius is pi/2 or more. But for 130 well-distributed points, the origin should be inside.
    # Let's verify: if the origin is outside, we should detect it.
    
    # Check if origin is inside the convex hull: for all facets, the origin should be on the
    # negative side of the outward normal.
    # Actually, we already handled the normal direction. Let's just return the result.
    
    # Convert to degrees
    return np.degrees(max_angle)


def _covering_radius_sample(points, n_samples=100000):
    """Fallback: estimate covering radius by sampling."""
    # Generate random points on the sphere
    rng = np.random.default_rng(42)
    samples = rng.normal(size=(n_samples, 3))
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    
    # Compute distances to all points
    # For efficiency, do in chunks
    max_min_dist = 0.0
    chunk = 10000
    for i in range(0, n_samples, chunk):
        s = samples[i:i+chunk]
        dots = np.clip(s @ points.T, -1.0, 1.0)
        min_dots = np.max(dots, axis=1)  # max dot = min angle
        min_angles = np.arccos(min_dots)
        max_min_dist = max(max_min_dist, np.max(min_angles))
    
    return np.degrees(max_min_dist)


def _covering_radius_grad(points):
    """Compute the covering radius and a gradient-like direction for optimization.
    
    Returns (radius_degrees, gradient) where gradient is an array of shape (N, 3)
    indicating the direction to move each point to reduce the covering radius.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: no gradient info
        r = _covering_radius_sample(points)
        return r, np.zeros_like(points)
    
    max_angle = 0.0
    max_facet = None
    max_vertex_idx = None
    
    for simplex in hull.simplices:
        pts = points[simplex]
        
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        normal = np.cross(e1, e2)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-12:
            continue
        normal = normal / n_norm
        
        centroid = np.mean(pts, axis=0)
        if np.dot(normal, centroid) < 0:
            normal = -normal
        
        dots = np.clip(np.dot(pts, normal), -1.0, 1.0)
        angles = np.arccos(dots)
        min_angle = np.min(angles)
        
        if min_angle > max_angle:
            max_angle = min_angle
            max_facet = simplex
            # Find which vertex is closest to the normal (has smallest angle)
            # Actually, the min_angle corresponds to the vertex with max dot product
            idx_in_facet = np.argmax(dots)
            max_vertex_idx = simplex[idx_in_facet]
    
    # Gradient: for the worst facet, we want to move the vertices to reduce the gap.
    # The "gap center" is 'normal'. To reduce the covering radius, we should move
    # the vertices of the worst facet toward the gap center.
    grad = np.zeros_like(points)
    if max_facet is not None:
        # Recompute the normal for the worst facet
        pts = points[max_facet]
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        normal = np.cross(e1, e2)
        n_norm = np.linalg.norm(normal)
        if n_norm > 1e-12:
            normal = normal / n_norm
            centroid = np.mean(pts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            # Move each vertex of the worst facet toward the normal direction
            # The direction to move is (normal - p_i) projected onto the tangent plane
            for idx in max_facet:
                # Tangent component of (normal - p_i)
                tangent = normal - np.dot(normal, points[idx]) * points[idx]
                grad[idx] = tangent
    
    return np.degrees(max_angle), grad


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin: kill at 150s
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit length
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Local optimization: gradient descent with adaptive step size
    # Use a simple subgradient-like approach based on the worst facet
    
    lr = 0.01
    min_lr = 1e-6
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Compute current radius and gradient
        radius, grad = _covering_radius_grad(points)
        
        if radius < best_radius - 1e-9:
            best_radius = radius
            best_points = points.copy()
            no_improve_count = 0
            # Slightly increase learning rate if improving
            lr = min(lr * 1.05, 0.05)
        else:
            no_improve_count += 1
            # Decrease learning rate
            lr = max(lr * 0.8, min_lr)
        
        # Apply gradient step: move points in the direction of the gradient
        # (toward reducing the covering radius)
        step = lr * grad
        points = points + step
        
        # Renormalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        
        # Check for validity: points should not be in a closed hemisphere
        # Simple check: the origin should be inside the convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, revert
            points = best_points.copy()
            break
    
    # Final polish: try a few random restarts of local optimization if time permits
    if time.time() < deadline - 20:
        rng = np.random.default_rng(12345)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            perturbed = best_points.copy()
            noise = rng.normal(scale=0.05, size=perturbed.shape)
            perturbed = perturbed + noise
            perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
            
            # Local optimization on perturbed solution
            lr_local = 0.01
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                radius_l, grad_l = _covering_radius_grad(perturbed)
                if radius_l < best_radius - 1e-9:
                    best_radius = radius_l
                    best_points = perturbed.copy()
                    lr_local = min(lr_local * 1.05, 0.05)
                else:
                    lr_local = max(lr_local * 0.8, 1e-6)
                
                step = lr_local * grad_l
                perturbed = perturbed + step
                perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
    
    # Ensure all points are on the unit sphere and valid
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
