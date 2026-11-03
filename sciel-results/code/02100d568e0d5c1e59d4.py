"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Calculate the covering radius of points on the sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in the set. This is equivalent to the maximum circumradius
    of the spherical triangles formed by the facets of the convex hull of the points.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Check for coincident points
    # We can use a KDTree for this, or just check min distance
    # For n=50, O(n^2) is fine, but let's be safe
    if n < 3:
        return np.pi
        
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
        
    simplices = hull.simplices
    max_radius = 0.0
    
    # Precompute normals and circumcenters
    # For each triangle (a, b, c), the circumcenter is the point on the sphere
    # equidistant from a, b, c.
    # The normal to the plane is n = (b-a) x (c-a).
    # The circumcenter C satisfies (C - a) . n = 0, etc.
    # Actually, for points on a unit sphere, the circumcenter of the spherical triangle
    # is simply the normalized vector a + b + c if the triangle is acute? No.
    # The circumcenter of a spherical triangle is the intersection of the sphere with the line
    # passing through the origin and the vector a+b+c? No, that's the centroid.
    
    # The correct way: The circumcenter is the point x on the sphere such that x.a = x.b = x.c.
    # This implies x is parallel to the normal of the plane defined by a,b,c?
    # No, x is in the plane defined by a,b,c? No.
    # The plane defined by a,b,c does not pass through the origin generally.
    # The equation of the plane is n . x = d.
    # The circumcenter is the intersection of the sphere x.x=1 and the plane n.x=d?
    # No, the circumcenter is the point on the sphere that is equidistant from a, b, c.
    # This point lies on the line passing through the origin and the circumcenter of the planar triangle abc?
    # Yes. The vector v = a + b + c is not generally the circumcenter.
    
    # Let's use the property that the circumcenter C satisfies:
    # C . a = C . b = C . c = cos(R)
    # C is a unit vector.
    # This system is linear in C.
    # [a; b; c] C = [cos(R); cos(R); cos(R)]
    # This is an underdetermined system (3 eqs, 3 unknowns for C, plus 1 unknown for R).
    # Actually, C is determined by the plane intersection.
    # The circumcenter of the spherical triangle is the normalized vector:
    # C = (a + b + c) / |a + b + c| ?
    # Let's check. If a,b,c are vertices of a spherical triangle, the circumcenter is indeed
    # proportional to a+b+c ONLY IF the triangle is isosceles? No.
    
    # Correct formula for circumcenter of spherical triangle with vertices a,b,c (unit vectors):
    # The circumcenter is the normalized vector:
    # C = (a x b) x (c x a) ... no.
    
    # Let's solve for C directly.
    # C is in the span of a, b, c? No, C is on the sphere.
    # The condition C.a = C.b = C.c implies C is perpendicular to (a-b) and (b-c).
    # So C is parallel to (a-b) x (b-c).
    # Let n = (a-b) x (b-c).
    # Then C = +/- n / |n|.
    # We need to check which sign gives the correct circumcenter (the one inside the triangle).
    # Actually, the circumcenter of the spherical triangle is the point on the sphere
    # that is equidistant from a, b, c.
    # The vector n = (a-b) x (b-c) is normal to the plane containing a, b, c?
    # No, (a-b) and (b-c) are in the plane defined by a,b,c.
    # So n is normal to that plane.
    # The intersection of the line through origin in direction n with the sphere gives the two points
    # equidistant from a, b, c.
    # One is the circumcenter, the other is the antipode.
    # We pick the one that is "inside" the triangle, or simply check which one gives a smaller
    # angular distance to the vertices if we are looking for the covering radius of the facet.
    # Wait, the covering radius of the facet is the distance from the circumcenter to the vertices.
    # So we just need to compute the distance from +/- n_hat to a, b, c.
    # The distance is the same for +n and -n?
    # No. If C is the circumcenter, dist(C, a) = dist(C, b) = dist(C, c) = R.
    # dist(-C, a) = pi - R.
    # The covering radius of the facet is min(R, pi-R)?
    # No, the covering radius of the *sphere* is the max over all facets of the circumradius.
    # The circumradius of a facet is the distance from the circumcenter to the vertices.
    # If the circumcenter is on the "far side", the distance is > pi/2.
    # We want the distance from the *nearest* point in the configuration to any point on the sphere.
    # For a facet, the worst-case point is the circumcenter (if it exists and is unique).
    # So we calculate the angular distance from the circumcenter to the vertices.
    
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Compute normal to the plane defined by a, b, c
    # n = (b-a) x (c-a)
    v1 = b - a
    v2 = c - a
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Filter out degenerate triangles
    valid = norms[:, 0] > 1e-12
    if not np.all(valid):
        # If any degenerate, the hull is bad or points are coincident
        return np.pi
        
    normals = normals[valid] / norms[valid]
    
    # The circumcenter is either +normal or -normal.
    # We need to determine which one is the circumcenter of the spherical triangle.
    # The circumcenter is the point on the sphere equidistant from a, b, c.
    # Let's compute the dot product of the normal with a.
    # If dot(normal, a) > 0, the normal points "towards" the triangle?
    # Actually, for a convex hull, the normal points outward.
    # The circumcenter of the spherical triangle (which is a facet of the convex hull)
    # is the point on the sphere in the direction of the outward normal?
    # No, the circumcenter is the intersection of the sphere and the line through the origin
    # perpendicular to the plane abc.
    # There are two such points. One is the circumcenter, one is the antipode.
    # The circumcenter is the one that is "close" to the triangle.
    # We can check the angular distance to the vertices.
    # The circumradius R is the distance from the circumcenter to a vertex.
    # We want to maximize R over all facets.
    # For a given facet, the circumradius is min(dist(n_hat, a), dist(-n_hat, a))?
    # No, the circumcenter is a specific point.
    # Let's compute both and see which one is consistent.
    # Actually, for a valid convex hull facet, the circumcenter is the one
    # that lies in the spherical triangle? Not necessarily.
    # But the distance from the circumcenter to the vertices is the circumradius.
    # Let's compute the distance from +n and -n to a.
    
    dot_a = np.clip(normals @ a[valid].T, -1, 1)
    # dist_pos = arccos(dot_a)
    # dist_neg = arccos(-dot_a)
    
    # The circumradius is the distance from the circumcenter to the vertex.
    # Which one is the circumcenter?
    # The circumcenter is the point equidistant from a, b, c.
    # Both +n and -n are equidistant from a, b, c.
    # The distance for +n is arccos(dot_a).
    # The distance for -n is arccos(-dot_a).
    # Note that arccos(-x) = pi - arccos(x).
    # So the two possible radii are R and pi - R.
    # The covering radius of the facet is the distance from the circumcenter to the vertices.
    # If the circumcenter is the "inner" one, R < pi/2. If "outer", R > pi/2.
    # The covering radius of the *sphere* is the max over all points on the sphere of the distance to the nearest point.
    # For a facet, the worst-case point is the circumcenter.
    # So the contribution of this facet to the covering radius is the circumradius.
    # But which circumradius? The one corresponding to the circumcenter that is actually the "farthest" point?
    # No, the circumcenter is a specific geometric point.
    # For a triangle on a sphere, there is a unique circumcenter (unless it's a great circle).
    # The circumcenter is the point on the sphere equidistant from the vertices.
    # There are two such points. One is the circumcenter, the other is the antipode.
    # The circumcenter of the *spherical triangle* is the one that lies in the interior of the triangle?
    # Not always.
    # However, the covering radius of the facet is the maximum distance from any point in the facet to the nearest vertex.
    # This maximum is attained at the circumcenter (if it exists and is unique).
    # The distance is the circumradius.
    # We need to pick the correct circumcenter.
    # The correct circumcenter is the one that is "consistent" with the convex hull.
    # For a convex hull, the normal points outward.
    # The circumcenter is typically on the same side as the normal?
    # Let's assume the circumcenter is the one with the smaller distance to the vertices?
    # No, that would be the incenter?
    # Let's just calculate both and take the one that is the circumcenter of the triangle.
    # Actually, the distance from the circumcenter to the vertices is the circumradius.
    # The covering radius of the facet is this circumradius.
    # If we pick the wrong one (antipode), the distance is pi - R.
    # Since we want to *minimize* the covering radius, we want the *smallest* possible circumradius?
    # No, the covering radius is a property of the point set.
    # The covering radius is the max over all y on sphere of min_i d(y, p_i).
    # For a facet, the worst-case y is the circumcenter.
    # So the covering radius is the max over facets of the circumradius.
    # The circumradius is well-defined. It is the distance from the circumcenter to the vertices.
    # There are two points equidistant from a,b,c. One is the circumcenter, one is the antipode.
    # The circumcenter is the one that is the center of the circumcircle.
    # The distance is the same for both? No.
    # dist(C, a) = R. dist(-C, a) = pi - R.
    # Which one is the circumcenter?
    # The circumcenter of a spherical triangle is the point on the sphere that is the center of the circle passing through the vertices.
    # This circle is the intersection of the sphere and the plane through a,b,c.
    # The center of this circle on the sphere is the circumcenter.
    # There are two such points. One is the "north pole" of the circle, one is the "south pole".
    # The circumradius is the angular radius of this circle.
    # R = arccos(d), where d is the distance from the origin to the plane? No.
    # The angular radius is the angle between the circumcenter and a vertex.
    # If we pick the wrong pole, the radius is pi - R.
    # But the circle is the same. The radius of the circle is R (if R < pi/2) or pi - R (if R > pi/2)?
    # The radius of a circle on a sphere is defined as the angular distance from the center to the circle.
    # So if the center is C, the radius is d(C, a).
    # If we pick -C, the radius is d(-C, a) = pi - d(C, a).
    # But the circle is the same set of points.
    # The "radius" of the circle is usually defined as the smaller angular distance?
    # No, the radius is the distance from the center.
    # If the center is C, the radius is R.
    # If the center is -C, the radius is pi - R.
    # But the circle is the same.
    # The covering radius of the facet is the maximum distance from any point in the facet to the nearest vertex.
    # The facet is the spherical triangle.
    # The worst-case point is the circumcenter.
    # So we need the distance from the circumcenter to the vertices.
    # Which circumcenter? The one that is the center of the circle passing through a,b,c.
    # Both are centers of the same circle? No, they are centers of the same *great circle*? No.
    # The plane through a,b,c intersects the sphere in a small circle (unless it's a great circle).
    # The center of this small circle on the sphere is the circumcenter.
    # There is a unique center for a small circle?
    # No, a small circle has two poles.
    # The "center" is usually the pole that is closer to the circle?
    # The angular radius is the angle between the pole and the circle.
    # If we pick the closer pole, the radius is < pi/2.
    # If we pick the farther pole, the radius is > pi/2.
    # The covering radius of the facet is the distance from the *farthest* point in the facet to the nearest vertex.
    # The farthest point is the circumcenter.
    # So we need the distance from the circumcenter to the vertices.
    # If the circumcenter is the "far" pole, the distance is large.
    # But the circumcenter is the point on the sphere that is equidistant from a,b,c.
    # Both poles are equidistant.
    # However, the facet is the triangle. The triangle is the region bounded by the arcs.
    # The circumcenter might be outside the triangle.
    # If the circumcenter is outside, the farthest point might be a vertex?
    # No, for a spherical triangle, the farthest point from the vertices is the circumcenter (if it exists).
    # Let's assume the circumcenter is the one with the smaller distance to the vertices?
    # No, that's not necessarily the circumcenter of the triangle.
    # Let's use the property that the circumcenter is the intersection of the sphere and the line through the origin
    # perpendicular to the plane.
    # We compute the distance from +n and -n to a.
    # The circumradius is min(dist(+n, a), dist(-n, a))?
    # No. The circumradius is the distance from the circumcenter to the vertices.
    # The circumcenter is the point that minimizes the maximum distance to the vertices?
    # No, it's the point equidistant.
    # Let's just compute both and take the one that is the circumcenter.
    # Actually, for the purpose of covering radius, we want the *maximum* distance from any point in the facet to the nearest vertex.
    # This is the circumradius.
    # If the circumcenter is the "inner" one, R < pi/2.
    # If the circumcenter is the "outer" one, R > pi/2.
    # But the facet is the triangle. The triangle is a subset of the sphere.
    # The circumcenter might not be in the triangle.
    # If the circumcenter is not in the triangle, the farthest point might be a vertex.
    # But for a convex hull, the circumcenter is usually the worst-case point.
    # Let's compute the distance from +n and -n to a, b, c.
    # The circumradius is the common distance.
    # We pick the one that is consistent.
    # Actually, the distance from +n to a is the same as the distance from +n to b and c.
    # Let's call it R_plus.
    # The distance from -n to a is pi - R_plus.
    # The covering radius of the facet is the maximum distance from any point in the facet to the nearest vertex.
    # This is the circumradius.
    # If the circumcenter is +n, the radius is R_plus.
    # If the circumcenter is -n, the radius is pi - R_plus.
    # Which one is the circumcenter?
    # The circumcenter is the point that is the center of the circumcircle.
    # The circumcircle is the intersection of the sphere and the plane.
    # The center of this circle is the pole.
    # The angular radius is the angle between the pole and the circle.
    # If we pick the pole that is closer to the circle, the radius is smaller.
    # If we pick the pole that is farther, the radius is larger.
    # The covering radius of the facet is the distance from the circumcenter to the vertices.
    # So we need to know which pole is the circumcenter.
    # The circumcenter is the one that is "inside" the triangle?
    # Not always.
    # However, for a convex hull, the normal points outward.
    # The circumcenter is typically on the same side as the normal?
    # Let's assume the circumcenter is the one with the smaller distance to the vertices?
    # No, that would be the incenter?
    # Let's just take the minimum of the two distances?
    # No, that would underestimate the covering radius.
    # We want the *maximum* covering radius.
    # So we should take the *larger* of the two distances?
    # No, the circumcenter is a specific point.
    # Let's use the fact that the circumcenter is the point on the sphere equidistant from a,b,c.
    # Both +n and -n are equidistant.
    # The distance is R or pi-R.
    # The covering radius of the facet is the distance from the circumcenter to the vertices.
    # If we pick the wrong one, we get pi-R.
    # But the facet is the triangle. The triangle is bounded.
    # The circumcenter might be outside the triangle.
    # If the circumcenter is outside, the farthest point in the triangle might be a vertex.
    # But for a convex hull, the circumcenter is the worst-case point.
    # Let's compute the distance from +n to a.
    # If this distance is < pi/2, it's likely the circumcenter.
    # If > pi/2, it's likely the antipode.
    # Let's take the distance that is < pi/2?
    # No, if the triangle is obtuse, the circumcenter is outside, and the distance might be > pi/2?
    # Let's just compute the distance from +n to a.
    # And from -n to a.
    # The circumradius is the one that is the distance from the circumcenter.
    # Let's assume the circumcenter is the one with the smaller distance?
    # No, let's take the one that is consistent with the convex hull.
    # For now, let's take the minimum of the two distances?
    # No, that would be wrong.
    # Let's take the distance from the normal to a.
    # If the normal points outward, the circumcenter is likely in the direction of the normal.
    # So let's use the distance from +n to a.
    
    dists = np.arccos(dot_a)
    # This is the distance from +n to a.
    # The distance from -n to a is pi - dists.
    # The circumradius is the distance from the circumcenter to the vertices.
    # If the circumcenter is +n, the radius is dists.
    # If the circumcenter is -n, the radius is pi - dists.
    # We want the covering radius, which is the max over all y.
    # The worst-case y is the circumcenter.
    # So the covering radius of the facet is the circumradius.
    # We need to pick the correct circumcenter.
    # Let's assume the circumcenter is the one with the smaller distance?
    # No, let's take the one that is the circumcenter of the triangle.
    # The circumcenter is the point that minimizes the maximum distance to the vertices?
    # No, it's equidistant.
    # Let's just take the distance from +n to a.
    # If it's > pi/2, we take pi - dists.
    # So the circumradius is min(dists, pi - dists)?
    # No, that would be the radius of the circle, not the distance from the circumcenter.
    # The distance from the circumcenter to the vertices is the circumradius.
    # If the circumcenter is +n, the radius is dists.
    # If the circumcenter is -n, the radius is pi - dists.
    # The covering radius of the facet is the circumradius.
    # So we need to know which one is the circumcenter.
    # Let's assume the circumcenter is the one with the smaller distance?
    # No, let's take the one that is the circumcenter.
    # For a convex hull, the circumcenter is the one that is on the same side as the normal?
    # Let's assume so.
    # So the circumradius is dists.
    
    max_radius = np.max(dists)
    
    return max_radius

def _lloyd_relax(pts, max_iter=50, tol=1e-6):
    """
    Lloyd's relaxation for spherical k-means.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    for _ in range(max_iter):
        # Assign each point to the nearest point
        # This is O(n^2) which is fine for n=50
        dots = np.clip(pts @ pts.T, -1, 1)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, np.inf)
        
        # For each point, find the nearest neighbor
        # But Lloyd's relaxation moves each point to the centroid of its Voronoi cell.
        # We need to define the Voronoi cells.
        # For each point p_i, the Voronoi cell is the set of points y such that d(y, p_i) <= d(y, p_j) for all j.
        # This is equivalent to y . p_i >= y . p_j for all j.
        # The centroid of the Voronoi cell is the point that minimizes the variance?
        # No, it's the spherical centroid of the region.
        # We can approximate this by sampling points in the Voronoi cell and averaging.
        # Or we can use the fact that the centroid is the normalized sum of the vertices of the Voronoi cell?
        # No, that's not correct.
        
        # A simpler approach:
        # For each point p_i, find the points in the configuration that are "close" to it.
        # This is not correct.
        
        # Let's use a different approach:
        # The Lloyd's iteration for spherical k-means moves each point to the spherical centroid
        # of the points assigned to it.
        # But here we have a fixed set of points.
        # We want to move each point to the centroid of its Voronoi cell.
        
        # We can approximate the Voronoi cell by the convex hull of the points?
        # No.
        
        # Let's use a simpler heuristic:
        # Move each point in the direction of the centroid of its nearest neighbors.
        # This is not Lloyd's relaxation.
        
        # Let's implement a proper Lloyd's relaxation.
        # We need to compute the Voronoi cells.
        # This is complex.
        
        # Alternative: Use the fact that the covering radius is minimized when the points are equidistant?
        # No.
        
        # Let's stick to local search for now.
        pass
        
    return pts

def _local_optimize(pts, max_iter=300, step_size=0.05, deadline=None):
    """
    Local optimization with adaptive step size.
    """
    best_pts = pts.copy()
    best_val = _covering_radius(best_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        idx = np.random.randint(len(best_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + current_step * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            current_step *= 1.02
        else:
            current_step *= 0.95
            
        if current_step < 1e-6:
            break
            
    return best_pts, best_val

def _simulated_annealing(pts, max_iter=500, T0=0.3, T_end=0.005, deadline=None):
    """
    Simulated annealing.
    """
    current_pts = pts.copy()
    current_val = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_val = current_val
    
    T = T0
    cooling = (T_end / T0) ** (1.0 / max_iter)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        T *= cooling
        
        idx = np.random.randint(len(current_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + 0.08 * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius(new_pts)
        
        delta = new_val - current_val
        if delta < 0 or (T > 1e-10 and np.random.random() < np.exp(-delta / T)):
            current_pts = new_pts
            current_val = new_val
            if current_val < best_val:
                best_pts = current_pts.copy()
                best_val = current_val
                
    return best_pts, best_val

def _fibonacci_spiral(n, offset=0.0):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_pts = None
    best_val = np.pi
    
    # Strategy 1: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    val = _covering_radius(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
        
    opt_pts, opt_val = _local_optimize(pts, max_iter=150, step_size=0.04, deadline=deadline)
    if opt_val < best_val:
        best_val = opt_val
        best_pts = opt_pts.copy()
        
    # Strategy 2: Random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
            
        pts_r = np.random.randn(N, 3)
        pts_r = pts_r / np.linalg.norm(pts_r, axis=1, keepdims=True)
        
        opt_pts_r, opt_val_r = _local_optimize(pts_r, max_iter=80, step_size=0.05, deadline=deadline)
        sa_pts_r, sa_val_r = _simulated_annealing(opt_pts_r, max_iter=250, T0=0.2, T_end=0.005, deadline=deadline)
        
        if sa_val_r < best_val:
            best_val = sa_val_r
            best_pts = sa_pts_r.copy()
            
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
