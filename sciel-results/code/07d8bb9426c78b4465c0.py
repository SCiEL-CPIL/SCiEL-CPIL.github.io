import numpy as np
from scipy.spatial import ConvexHull

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1)
    return points / norms[:, np.newaxis]

def _get_circumcenter(p1, p2, p3):
    """
    Compute the circumcenter of a spherical triangle defined by p1, p2, p3.
    The circumcenter is the point on the sphere equidistant from p1, p2, p3.
    It is proportional to the cross product of (p2 x p3) and (p3 x p1) etc?
    Actually, the normal to the plane containing p1, p2, p3 is n = (p2-p1) x (p3-p1).
    The circumcenter c is in the direction of n.
    c = n / ||n||
    """
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        return None
    return n / norm_n

def _angular_distance(p, q):
    dot = np.dot(p, q)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def construct():
    # Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # Local Optimization Loop
    # We will iterate, finding the worst facet and shrinking it.
    
    for iteration in range(5000):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        worst_radius = 0.0
        worst_idx = -1
        worst_circumcenter = None
        
        # Evaluate all facets
        radii = np.zeros(len(hull.simplices))
        centers = [None] * len(hull.simplices)
        
        for i, simplex in enumerate(hull.simplices):
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            c = _get_circumcenter(p1, p2, p3)
            if c is None:
                continue
            
            # The circumcenter might be on the opposite side of the sphere (if the triangle contains the antipode?)
            # For a convex hull of points on the unit sphere, the circumcenter of a facet should be the point
            # furthest from the origin in the direction of the normal.
            # However, we must ensure c is the one "outside" the hull.
            # Since the hull is convex and contains the origin (if points are in general position and not in a hemisphere),
            # the normal pointing outward is the one with positive dot product with the centroid?
            # Actually, for a triangle on the sphere, the circumcenter is unique up to sign.
            # The one relevant for covering is the one that is a local maximum of distance to the set.
            # Usually, the circumcenter of the spherical triangle is the one with positive dot product with the plane normal.
            
            # Check which side is the "empty" cap.
            # The covering radius is determined by the distance from the circumcenter to the vertices.
            # We need to ensure we pick the circumcenter that is NOT inside the convex hull?
            # Actually, for a facet of the convex hull, the circumcenter of the spherical triangle is the point
            # that maximizes the distance to the three vertices within the cap defined by the facet.
            
            # Let's compute the angular distance from c to p1.
            # If c is the "inner" circumcenter, the distance would be > pi/2?
            # No, the circumcenter is equidistant.
            # Let's just compute the distance.
            
            d1 = _angular_distance(c, p1)
            # Ensure we are looking at the cap. If d1 > pi/2, the circumcenter is on the other side?
            # Actually, if the triangle is large, the circumcenter might be far.
            # The covering radius is the max over all points. The worst point in a facet's cap is the circumcenter.
            # We must ensure the circumcenter is actually in the "empty" region.
            # If the dot product of c with the centroid of the triangle is positive, it's likely the correct one for the local cap.
            
            centroid = (p1 + p2 + p3) / 3.0
            if np.dot(c, centroid) < 0:
                c = -c
                d1 = _angular_distance(c, p1)
            
            radii[i] = d1
            centers[i] = c
            
            if d1 > worst_radius:
                worst_radius = d1
                worst_idx = i
                worst_circumcenter = c

        if worst_idx == -1:
            break
            
        # Move the vertices of the worst facet towards the circumcenter
        # This reduces the size of the triangle and thus its circumradius.
        simplex = hull.simplices[worst_idx]
        idxs = simplex
        
        # Step size
        step = 0.01
        
        new_points = points.copy()
        
        # Calculate target positions: move each vertex towards the circumcenter
        for idx in idxs:
            p = points[idx]
            # Move p towards worst_circumcenter
            # Vector from p to c
            # Project c onto tangent plane at p?
            # Or simply: p_new = normalize(p + step * (c - (p.dot(c)*p)))
            # Let's use a simple spherical interpolation or linear step projected.
            
            # Project c onto tangent plane of p
            proj = c - np.dot(c, p) * p
            new_points[idx] = p + step * proj
            
        # Normalize
        new_points = _normalize(new_points)
        
        # Check if this move improves the global covering radius
        # We need to recompute the hull and max radius.
        # To save time, we can just accept if it reduces the local worst radius significantly,
        # but it might increase others. 
        # Let's do a quick check: recompute hull and max radius.
        
        try:
            hull_new = ConvexHull(new_points)
            max_r_new = 0.0
            for simplex_new in hull_new.simplices:
                p1, p2, p3 = new_points[simplex_new[0]], new_points[simplex_new[1]], new_points[simplex_new[2]]
                c_new = _get_circumcenter(p1, p2, p3)
                if c_new is None:
                    continue
                centroid_new = (p1 + p2 + p3) / 3.0
                if np.dot(c_new, centroid_new) < 0:
                    c_new = -c_new
                d_new = _angular_distance(c_new, p1)
                if d_new > max_r_new:
                    max_r_new = d_new
            
            if max_r_new < worst_radius - 1e-6:
                points = new_points
        except Exception:
            pass
            
    return points
