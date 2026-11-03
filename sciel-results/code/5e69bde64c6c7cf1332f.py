"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START

def _smooth_max(x, sigma):
    """Soft max approximation."""
    # np.logaddexp.reduce is stable for max, but let's do a manual softmax for gradient control
    # Actually, for repulsive forces, we want the direction to the point that is 'most' responsible for the coverage gap.
    # A simple heuristic: find the point y on the sphere farthest from the set X.
    # This is equivalent to finding the vertex of the Voronoi cell with the largest circumradius.
    # We can approximate this by maximizing the distance.
    # Since we can't easily find the exact y, we can use a gradient ascent on y for a few steps, 
    # or use the property that the circumcenter of the spherical triangle formed by the 3 nearest points is the worst point.
    
    pass

def _refine(points, time_limit):
    """
    Iteratively refine points to minimize the maximum angular distance.
    Strategy:
    1. Find the point on the sphere that is farthest from the current configuration (worst coverage point).
    2. Move the configuration points slightly away from this worst point, or move the nearest points towards it?
       Actually, to reduce the max distance, we should move the points *towards* the worst point?
       No, if y is the worst point, it is the center of a "hole". The points surrounding it define the hole.
       Moving them *away* from each other opens the hole. Moving them *towards* y closes the hole?
       Wait. The covering radius is the distance from y to its nearest neighbor.
       If we move the nearest neighbors *towards* y, the distance decreases.
       However, this might increase the distance for other points.
       
       A standard approach for Tammes problem (max min distance) is repulsion.
       For Covering (min max distance), it's attraction to the holes.
       
       Let's use a "Lloyd's Algorithm" variant for spherical k-means but for covering:
       1. Find Voronoi cells (approx).
       2. Find the circumcenter of each cell (the point in the cell farthest from the seed).
       3. Move the seed towards the circumcenter.
       
       Finding exact spherical Voronoi cells is hard.
       Approximation:
       For each point p_i, find its 3 nearest neighbors.
       The "hole" is likely near the circumcenter of the triangle (p_i, p_j, p_k).
       Actually, the worst point y is the one that maximizes min(||y - p_i||).
       
       Let's try a simpler gradient-based approach:
       Objective: F(X) = max_y min_i angle(y, p_i)
       This is non-smooth.
       
       Alternative:
       Use a "Repulsive" model where points repel each other, but also "attract" to the empty space.
       
       Let's stick to a robust heuristic:
       1. Compute the covering radius and the worst point y.
       2. Identify the 3-4 points nearest to y.
       3. Move these points slightly towards y.
       4. Normalize.
       5. Repeat.
       
       To find y efficiently:
       The worst point y is the circumcenter of the spherical triangle formed by 3 points on the convex hull of the configuration that form a face with the largest circumradius.
       We can approximate this by:
       - For each point p_i, look at its neighbors.
       - Actually, we can just use a random search for y? No, too slow.
       
       Better:
       Use the fact that the covering radius is determined by the facets of the convex hull.
       The facet with the largest circumradius determines the covering radius.
       We can compute the convex hull of the 130 points.
       For each facet (triangle), compute the circumradius.
       Move the 3 vertices of the worst facet towards the circumcenter.
       
       This requires a convex hull algorithm. scipy.spatial.ConvexHull is available.
       
       Steps:
       1. Compute ConvexHull of points.
       2. For each simplex (facet), compute the circumcenter on the sphere.
       3. Compute the distance from the circumcenter to the vertices (circumradius).
       4. Find the facet with the maximum circumradius.
       5. Move the 3 vertices of this facet towards the circumcenter by a small step.
       6. Repeat until time runs out or convergence.
    """
    points = points.copy()
    start_time = time.time()
    
    # Helper to compute circumcenter of a spherical triangle
    def spherical_circumcenter(p1, p2, p3):
        # The circumcenter is the point equidistant from p1, p2, p3.
        # It lies on the normal to the plane of the triangle? No, on the sphere.
        # It is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle?
        # Yes, if the triangle is not containing the origin (which it doesn't, points are on sphere).
        # The plane of p1, p2, p3 has a normal n = (p1-p2) x (p1-p3).
        # The circumcenter C is k * n? No.
        # C is in the span of p1, p2, p3.
        # C = a p1 + b p2 + c p3 such that |C-p1|^2 = |C-p2|^2 = |C-p3|^2 = R^2 and |C|=1.
        
        # Simpler: The circumcenter of the spherical triangle is the normalized vector sum of the normals to the edges?
        # Actually, the circumcenter is the pole of the plane passing through p1, p2, p3?
        # No, the pole is the center of the circle *if* the triangle is right-angled or something?
        
        # Let's use the linear algebra solution.
        # Minimize |C|^2 subject to C dot (p1-p2) = (|p1|^2 - |p2|^2)/2 = 0 (since |p|=1)
        # So C is orthogonal to (p1-p2) and (p1-p3).
        # Thus C is parallel to (p1-p2) x (p1-p3).
        
        n = np.cross(p1 - p2, p1 - p3)
        if np.linalg.norm(n) < 1e-10:
            return None # Degenerate
        c = n / np.linalg.norm(n)
        # Check if c is on the same side as the points?
        # The circumcenter might be the antipode if the triangle is large.
        # We want the circumcenter that is the center of the empty cap.
        # The empty cap is the one that doesn't contain other points?
        # For the convex hull, the facet is the one facing outward.
        # The normal of the facet (outward) is n_out.
        # The circumcenter of the facet is along n_out?
        # Yes, the circumcenter of a spherical triangle is the point on the sphere along the normal to the plane of the triangle.
        # Wait, is it?
        # Let p1, p2, p3 be on the unit sphere.
        # The plane is a.x + b.y + c.z = d.
        # The normal is (a,b,c).
        # The circumcenter is (a,b,c)/sqrt(a^2+b^2+c^2) ?
        # Let's check. Distance from C to p1: |C - p1|^2 = 1 + 1 - 2 C dot p1.
        # If C is parallel to normal n, C = n/|n|.
        # Does n dot p1 = n dot p2 = n dot p3?
        # n is normal to the plane containing p1, p2, p3. So yes, n dot p is constant for all p in the plane.
        # So C = n/|n| is equidistant from p1, p2, p3.
        # There are two such points: C and -C.
        # We want the one that is the center of the "hole", i.e., the one farthest from the origin in the direction of the facet?
        # The facet is part of the convex hull. The origin is inside the hull.
        # The normal n pointing away from the origin is the one we want.
        # Since the origin is inside, n dot 0 < n dot p1? No.
        # The plane equation is n dot x = h. Since 0 is inside, and points are on boundary, h > 0 if n is outward.
        # So we want the sign of n such that n dot p1 > 0.
        
        if np.dot(n, p1) < 0:
            c = -c
        return c

    # Initial refinement loop
    # We will perform a fixed number of iterations or until time runs out.
    # Each iteration:
    # 1. Convex Hull
    # 2. Find worst facet
    # 3. Update vertices
    
    # To make it faster, we can batch updates or use a smaller step.
    # Also, we should handle the case where the hull changes.
    
    # Let's run a loop for up to 100 iterations or 100 seconds.
    for _ in range(100):
        if time.time() - start_time > time_limit - 5: # 5s safety margin
            break
            
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(points)
            # hull.simplices contains indices of points forming each facet
            # For 3D points, simplices are triangles (size 3)
            
            max_rad = 0
            worst_facet = None
            worst_circumcenter = None
            
            for simplex in hull.simplices:
                p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
                c = spherical_circumcenter(p1, p2, p3)
                if c is None:
                    continue
                # Compute circumradius (angular distance)
                # angle = arccos(c dot p1)
                dot_val = np.dot(c, p1)
                dot_val = np.clip(dot_val, -1.0, 1.0)
                rad = np.arccos(dot_val)
                
                if rad > max_rad:
                    max_rad = rad
                    worst_facet = simplex
                    worst_circumcenter = c
                    
            if worst_facet is None:
                break
                
            # Move the 3 points towards the circumcenter
            # Step size: adaptive or fixed?
            # Let's use a small fixed step, e.g., 0.01 radians?
            # Or move them by a fraction of the distance?
            # Let's move them 10% of the way towards the circumcenter.
            step_size = 0.1
            for idx in worst_facet:
                p = points[idx]
                # Vector from p to c
                # Geodesic step
                # p_new = (p + step * (c - p)) / || ... || ?
                # Better: slerp
                # But simple linear interpolation in R3 followed by normalization works well for small steps
                p_new = p + step_size * (worst_circumcenter - p)
                norm = np.linalg.norm(p_new)
                if norm > 1e-10:
                    points[idx] = p_new / norm
        except Exception:
            # If hull fails (e.g. coplanar points), break or skip
            break
            
    return points

def construct():
    # Golden-angle (Fibonacci) spiral points on the unit sphere: a good but improvable start.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Refine using the strategy above
    # Budget: 100 seconds (limit is 120, keep margin)
    points = _refine(points, 100.0)
    
    return points

# EVOLVE-BLOCK-END
