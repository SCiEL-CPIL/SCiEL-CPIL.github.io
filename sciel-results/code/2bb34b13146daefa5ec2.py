"""
Optimized sphere covering with 130 points via Direct Geometric Facet Shrinkage.

Strategy:
1. Directly target the covering radius by computing the convex hull.
2. Identify the facet (triangle) with the largest circumradius.
3. Move the vertices of that facet towards the circumcenter of the facet to shrink it.
4. To escape local minima where the hull topology is stable but suboptimal,
   occasionally perturb points towards the "holes" (circumcenters of adjacent facets)
   or perform a random kick.
5. Use a simulated annealing-like schedule for the step size.

This is materially different from force-directed methods because it operates
directly on the constraint (covering radius) rather than a proxy energy (like
Thomson energy or spectral variance).
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 145  # 5s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=2.0):
    return time.time() < _DEADLINE - margin

def _compute_circumcenter_and_radius(a, b, c):
    """
    Compute the circumcenter and circumradius of a triangle on the sphere.
    Returns (circumcenter, radius_in_radians).
    Handles degenerate cases by returning the centroid and a large radius.
    """
    # Vectors
    v1 = b - a
    v2 = c - a
    
    # Cross product gives normal to the plane
    cross = np.cross(v1, v2)
    cnorm = np.linalg.norm(cross)
    
    if cnorm < 1e-10:
        # Degenerate: return centroid
        center = (a + b + c) / 3.0
        center = center / np.linalg.norm(center)
        # Radius is max distance to vertices
        r = max(np.arccos(np.clip(np.dot(center, a), -1.0, 1.0)),
                np.arccos(np.clip(np.dot(center, b), -1.0, 1.0)),
                np.arccos(np.clip(np.dot(center, c), -1.0, 1.0)))
        return center, r

    # Normalized normal vector
    n = cross / cnorm
    
    # The circumcenter lies on the line passing through the origin and n
    # (since the triangle is on the unit sphere, the circumcenter is along the normal)
    # Actually, the circumcenter C satisfies |C|^2 = 1 and (C-a)^2 = (C-b)^2 = (C-c)^2.
    # This implies C is in the span of a, b, c.
    # Specifically, C = alpha * n + beta * (some vector in plane)?
    # Simpler: C = k * n.
    # Distance from C to a: |k*n - a|^2 = k^2 - 2k(n.a) + 1 = R^2
    # We want C to be on the sphere, so k=1 or k=-1.
    # Wait, the circumcenter of a spherical triangle is the intersection of the great circle bisectors.
    # For a triangle on the unit sphere, the circumcenter is indeed along the normal vector n.
    # Let C = sign * n.
    # The radius r is the angular distance from C to a.
    # cos(r) = C . a = sign * (n . a)
    
    # We need to choose the sign such that C is on the same side as the triangle centroid?
    # Or simply, we want the cap containing the triangle to be minimal.
    # The circumcenter is the point on the sphere equidistant from a, b, c.
    # There are two such points: n and -n.
    # One is the "north" pole, one is the "south" pole relative to the triangle plane.
    # The covering radius is determined by the facet. The facet is the triangle itself.
    # The "cap" defined by the facet is the smaller cap containing the triangle.
    # So we pick the normal that points towards the interior of the convex hull?
    # No, the facet is on the boundary. The interior is in the direction of the centroid of the hull?
    # Actually, for the covering radius, we care about the empty cap.
    # The empty cap is the one that does NOT contain the origin? No, the origin is inside the hull.
    # The empty cap is the one that is "outside" the hull?
    # Let's check the dot product with the centroid of the triangle.
    centroid = (a + b + c) / 3.0
    if np.dot(n, centroid) < 0:
        n = -n
        
    # Now n points roughly towards the triangle.
    # The circumcenter is n (or -n?).
    # Let's test both.
    c1 = n
    c2 = -n
    
    r1 = np.arccos(np.clip(np.dot(c1, a), -1.0, 1.0))
    r2 = np.arccos(np.clip(np.dot(c2, a), -1.0, 1.0))
    
    # The covering radius for this facet is the max distance from the circumcenter to the vertices.
    # Since it's a circumcenter, distances are equal.
    # We want the "empty" cap. The empty cap is the one that doesn't contain other points.
    # Usually, for a convex hull facet, the outward normal points away from the interior.
    # The interior points are "below" the plane. The outward normal points "up".
    # The circumcenter of the facet (as a spherical triangle) is the pole.
    # If we pick the pole that is "outside" the hull, the cap is the one we want to minimize?
    # No, the covering radius is the max distance from ANY point on the sphere to the nearest node.
    # The worst point for a facet is its circumcenter.
    # The distance is the angular distance from the circumcenter to the vertices.
    # So we just need the correct circumcenter.
    
    # Let's verify which one is the circumcenter of the spherical triangle.
    # The spherical triangle is the region bounded by the arcs.
    # The circumcenter is inside this region.
    # The centroid is inside the region.
    # So the circumcenter should be close to the centroid.
    if np.dot(c1, centroid) > np.dot(c2, centroid):
        center = c1
        radius = r1
    else:
        center = c2
        radius = r2
        
    return center, radius

def _covering_radius_and_worst_facet(pts):
    """
    Compute covering radius and identify the worst facet (index in hull.simplices).
    Returns (max_radius, worst_facet_idx, worst_circumcenter).
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, -1, None

    max_r = 0.0
    worst_idx = -1
    worst_center = None

    for idx, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        center, r = _compute_circumcenter_and_radius(a, b, c)
        if r > max_r:
            max_r = r
            worst_idx = idx
            worst_center = center

    return max_r, worst_idx, worst_center

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    start_time = time.time()
    best_pts = None
    best_r = np.pi
    
    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415]:
        inits.append(_fibonacci_spiral(N, jitter=0.1, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    
    # Sort by initial radius
    init_scores = []
    for p in inits:
        r, _, _ = _covering_radius_and_worst_facet(p)
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])
    
    # Optimization Loop
    # We use a "shrink the worst facet" strategy.
    # To avoid getting stuck, we also randomly perturb points.
    
    rng = np.random.RandomState(12345)
    
    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break
            
        pts = p0.copy()
        current_r, worst_idx, worst_center = _covering_radius_and_worst_facet(pts)
        
        if current_r < best_r:
            best_r = current_r
            best_pts = pts.copy()
            
        # Annealing parameters
        T = 0.1
        T_min = 0.001
        alpha = 0.995
        
        max_iters = 5000
        
        for it in range(max_iters):
            if not _time_ok(5.0):
                break
                
            # Recompute hull and worst facet
            current_r, worst_idx, worst_center = _covering_radius_and_worst_facet(pts)
            
            if current_r < best_r:
                best_r = current_r
                best_pts = pts.copy()
                
            if worst_idx == -1:
                break
                
            # Get the vertices of the worst facet
            hull = ConvexHull(pts) # Re-computing hull is expensive, but we need simplices
            # Optimization: We already have hull from the function? No, we returned only values.
            # Let's modify _covering_radius... to return hull or re-compute.
            # For speed, let's assume we can access hull.simplices[worst_idx].
            # Actually, _covering_radius_and_worst_facet creates a new hull.
            # We should return the hull or the simplex.
            
            # Let's refactor _covering_radius_and_worst_facet to return the simplex too.
            # Wait, I can just re-run ConvexHull here. It's O(N log N).
            # N=130 is small.
            
            simplex = hull.simplices[worst_idx]
            v_idx = simplex
            
            # Move these vertices towards the circumcenter
            # Step size: proportional to the radius we want to remove?
            # We want to reduce the radius.
            # Moving towards the center reduces the distance to the center.
            # But it might increase distances to other points.
            
            # Heuristic step size: 
            # If we move a point P towards C by delta, the new distance to C is r - delta (approx).
            # We want to be careful not to overshoot or create new larger facets.
            
            step = 0.05 * T  # Small step
            
            for i in v_idx:
                # Move towards center
                pts[i] = pts[i] + step * (worst_center - pts[i])
                
            # Random perturbation to escape local minima
            if rng.rand() < 0.1:
                # Pick a random point and kick it
                k = rng.randint(N)
                pts[k] = pts[k] + 0.05 * T * rng.randn(3)
                
            pts = _normalize(pts)
            
            # Cool down
            T *= alpha
            if T < T_min:
                T = T_min
                
    # Final Polish: A few iterations of pure facet shrinking without random kicks
    if best_pts is not None and _time_ok(5.0):
        pts = best_pts.copy()
        T = 0.01
        for _ in range(200):
            if not _time_ok(2.0):
                break
            current_r, worst_idx, worst_center = _covering_radius_and_worst_facet(pts)
            if worst_idx == -1:
                break
            hull = ConvexHull(pts)
            simplex = hull.simplices[worst_idx]
            step = 0.01 * T
            for i in simplex:
                pts[i] = pts[i] + step * (worst_center - pts[i])
            pts = _normalize(pts)
            
            new_r, _, _ = _covering_radius_and_worst_facet(pts)
            if new_r < best_r:
                best_r = new_r
                best_pts = pts.copy()
            T *= 0.99

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    r, _, _ = _covering_radius_and_worst_facet(pts)
    print("Covering radius (deg):", np.degrees(r))
    print("Time: %.1f s" % (time.time() - t0))
