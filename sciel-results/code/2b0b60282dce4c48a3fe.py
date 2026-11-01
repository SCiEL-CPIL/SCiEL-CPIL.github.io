"""Improved program for covering the sphere with 130 equal caps."""
import numpy as np
import time

N = 130

def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of unit vectors.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest chosen point. This equals the maximum circumradius of the facets
    of the convex hull, where each facet's circumcenter is the farthest point from
    all hull vertices in that facet.
    
    We compute this by finding, for each point, the maximum distance to that point
    among all sphere points, but more accurately, we need the max over the sphere.
    
    A practical approach: the covering radius is determined by the "worst" region.
    For each pair/triple of nearby points, we can find the circumcenter of the 
    spherical triangle and check if it's equidistant to all three and farther from
    any other point. But a simpler robust method: sample the sphere densely and 
    compute max min-distance. However, for precision, we use the fact that the
    covering radius equals max over all points y on sphere of min_i angle(y, x_i).
    
    The critical points are the circumcenters of spherical triangles formed by
    neighboring points. We can approximate by: for each point, find its 3-4 nearest
    neighbors, compute the "hole" center, and measure distance.
    
    A robust numerical approach: use the convex hull. The covering radius is the 
    max over facets of the circumradius of the facet (in angular terms). For a 
    spherical polygon facet with vertices v1..vk, the circumcenter c is the point
    on the sphere equidistant from all vi (the normalized sum of vi, or solved).
    The angular distance from c to each vi is the circumradius of that facet.
    The covering radius is the max of these over all facets.
    
    However, computing the convex hull of 130 points in 3D and then for each facet
    finding the circumcenter is doable. Let's use scipy.spatial.ConvexHull.
    """
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(points)
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        # simplex contains 3 indices (triangular facet in 3D)
        tri = points[simplex]
        # Find circumcenter: the point on the sphere equidistant from all 3 vertices
        # For 3 points on the unit sphere, the circumcenter is the normalized 
        # solution to |c - v1| = |c - v2| = |c - v3| with |c| = 1
        # This means c · v1 = c · v2 = c · v3 = cos(theta)
        # So c is in the direction of the vector that satisfies these constraints.
        
        # The circumcenter of a spherical triangle can be found as follows:
        # It's the normalized vector in the direction of:
        #   v1 x v2 cross something... 
        # Actually, the circumcenter c satisfies c·v1 = c·v2 = c·v3.
        # So c is perpendicular to (v1 - v2) and (v1 - v3), i.e., c is parallel to
        #   (v1 - v2) x (v1 - v3)
        # But we also need |c| = 1.
        
        v1, v2, v3 = tri
        # c is parallel to (v1 - v2) x (v1 - v3)
        cross = np.cross(v1 - v2, v1 - v3)
        norm = np.linalg.norm(cross)
        if norm < 1e-12:
            # Degenerate facet
            continue
        c = cross / norm
        
        # Check which side: c should be on the same side as the origin relative to the facet
        # The facet normal from the hull points outward. The circumcenter should be
        # on the sphere on the "far" side from the origin, in the direction away from 
        # the bulk of the points.
        # Actually, for a convex hull of points on the unit sphere, the circumcenter
        # of a facet (as the center of the cap) is in the direction of the facet normal
        # pointing away from the origin (i.e., outward).
        
        # The hull's facet normal:
        # For a triangle with vertices v1, v2, v3 in CCW order as seen from outside,
        # the outward normal is (v2 - v1) x (v3 - v1) normalized.
        normal = np.cross(v2 - v1, v3 - v1)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            continue
        normal = normal / normal_norm
        
        # The circumcenter direction should be along the normal (outward)
        # Let's check: if c · normal < 0, flip c
        if np.dot(c, normal) < 0:
            c = -c
        
        # Now c is the circumcenter direction. The angular distance from c to v1:
        dot_val = np.dot(c, v1)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        angular_dist = np.arccos(dot_val)
        
        max_circumradius = max(max_circumradius, angular_dist)
    
    return max_circumradius


def _soft_covering_radius(points, temperature=0.05):
    """Compute a smooth surrogate for the covering radius using soft-min.
    
    For each point on the sphere, the distance to the nearest point is min_i angle(y, x_i).
    The covering radius is max_y min_i angle(y, x_i).
    
    A smooth surrogate: for a set of candidate worst-case points y, compute
    softmin_i angle(y, x_i) and take the max. But we don't have a grid of y.
    
    Alternative: use the convex hull approach but smooth it.
    
    For optimization, a simpler approach: minimize the sum of squared distances
    from the "holes" to their nearest points, where holes are approximated.
    
    Actually, let's use a different smooth objective: the sum over all pairs of
    points of a repulsive potential, plus a term that penalizes large gaps.
    
    For now, let's just use the exact covering radius for the final evaluation,
    and for optimization use a gradient-based approach on the points directly.
    """
    return _covering_radius(points)


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Gradient descent to minimize covering radius
    # We'll use a simple approach: for each point, compute a gradient based on
    # which "holes" it's responsible for, and move it to reduce the max circumradius.
    
    # Actually, let's use a more robust approach: simulated annealing or
    # local search with random perturbations.
    
    # Method: iteratively perturb each point in a small random direction,
    # normalize, and accept if the covering radius decreases.
    
    rng = np.random.RandomState(42)
    
    for it in range(5000):
        if time.time() > deadline - 10:
            break
        
        # Pick a random point to perturb
        idx = rng.randint(N)
        
        # Generate a random perturbation direction
        perturbation = rng.randn(3)
        perturbation /= np.linalg.norm(perturbation)
        
        # Try different step sizes
        improved = False
        for step in [0.05, 0.02, 0.01, 0.005]:
            new_points = points.copy()
            new_points[idx] = points[idx] + step * perturbation
            new_points[idx] /= np.linalg.norm(new_points[idx])
            
            new_radius = _covering_radius(new_points)
            
            if new_radius < best_radius - 1e-10:
                points = new_points
                best_points = points.copy()
                best_radius = new_radius
                improved = True
                break
        
        if not improved:
            # Try a smaller perturbation
            for step in [0.002, 0.001]:
                new_points = points.copy()
                new_points[idx] = points[idx] + step * perturbation
                new_points[idx] /= np.linalg.norm(new_points[idx])
                
                new_radius = _covering_radius(new_points)
                
                if new_radius < best_radius - 1e-10:
                    points = new_points
                    best_points = points.copy()
                    best_radius = new_radius
                    break
    
    # Final polish: try to improve further with very small steps
    for it in range(2000):
        if time.time() > deadline - 5:
            break
        
        idx = rng.randint(N)
        perturbation = rng.randn(3)
        perturbation /= np.linalg.norm(perturbation)
        
        for step in [0.001, 0.0005, 0.0002]:
            new_points = best_points.copy()
            new_points[idx] = best_points[idx] + step * perturbation
            new_points[idx] /= np.linalg.norm(new_points[idx])
            
            new_radius = _covering_radius(new_points)
            
            if new_radius < best_radius - 1e-10:
                best_points = new_points
                best_radius = new_radius
                break
    
    return best_points
