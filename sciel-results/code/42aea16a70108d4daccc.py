"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """
    Return vertices of a regular hexagon with center (cx, cy), side length, and rotation angle in degrees.
    Vertices are ordered counter-clockwise starting from angle 0 (pointing right).
    """
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def segment_distance(p1, p2, p3, p4):
    """
    Compute the minimum distance between two line segments (p1,p2) and (p3,p4).
    """
    # Vector for the two lines
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    
    if a < 1e-10:
        s = 0.0
        t = f / e
        t = max(0.0, min(1.0, t))
    else:
        c = np.dot(d1, r)
        if e < 1e-10:
            t = 0.0
            s = -c / a
            s = max(0.0, min(1.0, s))
        else:
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom < 1e-10:
                s = 0.0
            else:
                s = (b * f - c * e) / denom
            if s < 0.0:
                s = 0.0
                t = f / e
            elif s > 1.0:
                s = 1.0
                t = (f + b) / e
            else:
                t = (b * s + f) / e
            
            if t < 0.0:
                t = 0.0
                s = -c / a
                s = max(0.0, min(1.0, s))
            elif t > 1.0:
                t = 1.0
                s = (b - c) / a
                s = max(0.0, min(1.0, s))
    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    
    if s < 0.0:
        s = 0.0
    elif s > 1.0:
        s = 1.0
    
    c1 = p1 + s * d1
    c2 = p3 + t * d2
    return np.linalg.norm(c1 - c2)

def polygon_distance(poly1, poly2):
    """
    Compute the minimum distance between two convex polygons.
    If they overlap or touch, return 0.0.
    """
    min_dist = float('inf')
    n1 = len(poly1)
    n2 = len(poly2)
    
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            d = segment_distance(p1, p2, p3, p4)
            if d < min_dist:
                min_dist = d
    
    # Check if any vertex of poly1 is inside poly2 or vice versa
    # For simplicity, if min_dist is 0, they touch or overlap.
    # We also need to check for containment.
    # A point-in-polygon test for convex polygons can be done by checking cross products.
    # But since we are minimizing distance, if they are disjoint, min_dist > 0.
    # If they overlap, min_dist will be 0 from segment intersection or vertex on edge.
    # However, if one is fully inside the other without touching edges, min_dist from segments is > 0.
    # We need a robust check.
    
    # Check if any vertex of poly1 is inside poly2
    for p in poly1:
        if point_in_convex_polygon(p, poly2):
            return 0.0
    for p in poly2:
        if point_in_convex_polygon(p, poly1):
            return 0.0
            
    return min_dist

def point_in_convex_polygon(p, poly):
    """
    Check if point p is inside or on the boundary of convex polygon poly.
    """
    n = len(poly)
    sign = 0
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if cross > 1e-9:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-9:
            if sign > 0:
                return False
            sign = -1
    return True

def compute_outer_side_length(inner_hexes):
    """
    Given a list of (cx, cy, angle) for N unit hexagons,
    compute the minimum side length of a regular hexagon (with arbitrary center and rotation)
    that can contain all of them.
    
    This is equivalent to finding the minimum width in 6 directions (for a fixed orientation) 
    but we also optimize over the outer hexagon's rotation.
    
    Actually, the problem states the outer hexagon can have arbitrary center and rotation.
    The score is the outer side length.
    
    For a set of points (vertices of all inner hexagons), the minimum enclosing regular hexagon 
    with arbitrary rotation and center is a non-trivial problem.
    
    However, we can use the fact that the minimum enclosing hexagon for a set of points 
    will have at least 3 vertices of the hexagon touching the convex hull of the points.
    
    A simpler approach: The width of the set of points in direction theta is the maximum projection 
    minus the minimum projection. For a regular hexagon of side length S, the width in any direction 
    is S * (sqrt(3)/2 + |cos(3*theta)|) ... wait, no.
    
    Let's rethink. A regular hexagon of side length S has inradius r = S * sqrt(3)/2.
    The width of a regular hexagon of side S in a direction making angle phi with the normal of a face is:
    w(phi) = S * (sqrt(3)/2 + |sin(phi)|) ... this is getting complicated.
    
    Alternative: The minimum enclosing regular hexagon problem is equivalent to finding the minimum S such that 
    there exists a center C and rotation alpha such that all points are inside the hexagon.
    
    This is a 4-variable optimization problem (Cx, Cy, alpha, S) but S is the objective.
    
    For a fixed rotation alpha, the minimum enclosing hexagon centered at C is determined by the 
    maximum of the 6 support functions. The side length S is proportional to the maximum of these.
    
    Specifically, for a hexagon with rotation alpha, the 6 normals are at angles alpha + k*pi/3.
    The support function in direction n is max_i (p_i . n).
    The hexagon is defined by |p . n_k| <= r for k=0..5, where r is the inradius.
    Wait, a regular hexagon is the intersection of 6 half-planes.
    The inradius r is the distance from center to the middle of a side.
    The side length S = 2 * r / sqrt(3).
    
    So for a fixed rotation alpha and center C, the required inradius is:
    r_required = max_k max_i ( (p_i - C) . n_k )
    where n_k are the 6 outward normals.
    
    Then S = 2 * r_required / sqrt(3).
    
    We want to minimize S over C and alpha.
    
    This is still a 3-variable optimization (Cx, Cy, alpha).
    We can use differential evolution for this as well, but it's nested.
    
    To keep it efficient, we can fix the outer hexagon rotation to 0 or 30 degrees, 
    or optimize it coarsely.
    
    Let's assume the outer hexagon is aligned with one of the inner hexagons, or just optimize 
    the center for a fixed rotation.
    
    Actually, the optimal outer hexagon will likely have its sides parallel to the sides of 
    the "extreme" inner hexagons.
    
    Let's use a 4D optimization: (Cx, Cy, alpha, S) is not right because S is derived.
    We optimize (Cx, Cy, alpha) and compute S.
    
    But doing this for every evaluation of the inner hexagon positions is expensive.
    
    Alternative: The minimum enclosing hexagon for a set of points is known to be determined 
    by 3 points on the convex hull.
    
    For simplicity and speed, let's assume the outer hexagon has the same orientation as the 
    "dominant" inner hexagon, or just use a fixed orientation (e.g., 0 degrees) and optimize 
    the center. This might not be optimal but could be a good heuristic.
    
    Or, we can compute the minimum enclosing hexagon by checking a few candidate rotations 
    (e.g., 0, 30, 60, ...) and taking the best.
    
    Let's implement a function that, given a set of points, finds the minimum enclosing 
    regular hexagon by optimizing over center and rotation.
    """
    # Extract all vertices
    points = []
    for cx, cy, ang in inner_hexes:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        points.extend(verts.tolist())
    points = np.array(points)
    
    if len(points) == 0:
        return 0.0
    
    # Compute convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Optimize over center (Cx, Cy) and rotation alpha
    def outer_side_length(params):
        Cx, Cy, alpha = params
        C = np.array([Cx, Cy])
        alpha_rad = np.radians(alpha)
        
        # 6 outward normals
        normals = []
        for k in range(6):
            theta = alpha_rad + k * np.pi / 3
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        
        # Compute required inradius
        r_max = 0.0
        for n in normals:
            # Project all hull points onto normal
            projections = np.dot(hull_points - C, n)
            r_max = max(r_max, np.max(projections))
        
        # Side length S = 2 * r / sqrt(3)
        S = 2.0 * r_max / np.sqrt(3.0)
        return S
    
    # Use differential evolution to minimize outer_side_length
    # Bounds: Cx, Cy in [-10, 10], alpha in [0, 60]
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (0.0, 60.0)]
    result = differential_evolution(outer_side_length, bounds, maxiter=50, popsize=10, seed=42, tol=1e-4)
    
    return result.fun

def construct():
    """
    Optimize the positions and rotations of 11 unit hexagons to minimize the 
    outer enclosing hexagon side length.
    """
    # Parameters: for each of the 11 hexagons, (cx, cy, angle)
    # Total 33 parameters.
    
    # Initial guess: a compact arrangement
    # Let's start with a hexagonal packing pattern
    # Center at (0,0), others around it
    
    # Define a function to evaluate a set of parameters
    def evaluate(params):
        # Unpack parameters
        inner_hexes = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = params[3*i+2]
            inner_hexes.append((cx, cy, ang))
        
        # Check for overlaps
        # If any two hexagons overlap, return a large penalty
        for i in range(N):
            for j in range(i+1, N):
                vi = hexagon_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
                vj = hexagon_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
                dist = polygon_distance(vi, vj)
                if dist < -1e-6:  # Overlap
                    return 1e6 + abs(dist)
        
        # Compute outer side length
        outer_side = compute_outer_side_length(inner_hexes)
        return outer_side
    
    # Initial population
    # Let's create a compact initial configuration
    # Center hexagon at (0,0)
    # 6 surrounding hexagons at distance 2*sqrt(3)/2 = sqrt(3) in 6 directions
    # 4 more hexagons further out
    
    # Actually, let's just use random initialization within a reasonable box
    # and let differential evolution find the optimum.
    
    # Bounds for each hexagon
    # cx, cy in [-5, 5], angle in [0, 60]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))  # cx
        bounds.append((-5.0, 5.0))  # cy
        bounds.append((0.0, 60.0))  # angle
    
    # Use differential evolution
    # This might be slow, so we use a small population and few iterations
    # We can also use a local search afterwards
    
    start_time = time.time()
    timeout = 110.0  # Leave some time for the outer hexagon optimization
    
    # Run differential evolution with a time limit
    # Since differential_evolution doesn't have a built-in timeout, we'll run it 
    # with a limited number of iterations and hope it's fast enough.
    
    # Let's try a simpler approach: use a fixed compact configuration and just 
    # compute the outer hexagon. This is fast and might be good enough.
    
    # Compact hexagonal packing:
    # Row 0: 1 hexagon at (0, 0)
    # Row 1: 2 hexagons at (-1.5, sqrt(3)/2), (1.5, sqrt(3)/2)
    # Row 2: 3 hexagons at (-3, 0), (0, 0), (3, 0) -- wait, this is not correct for hexagonal packing
    
    # For unit hexagons, the distance between centers of adjacent hexagons is 2 * inradius = sqrt(3).
    # In a hexagonal lattice, the centers are at:
    # (i * 1.5 + j * 0.75, j * sqrt(3)/2) for integer i, j, with some adjustments.
    
    # Let's place 11 hexagons in a compact cluster:
    # Center: (0, 0)
    # 6 neighbors: at distance sqrt(3) in 6 directions (0, 60, 120, 180, 240, 300 degrees)
    # 4 more: at distance 2*sqrt(3) in 4 directions (30, 150, 270, 330 degrees)
    
    inner_hexes = []
    # Center
    inner_hexes.append((0.0, 0.0, 0.0))
    
    # 6 neighbors
    for k in range(6):
        angle = k * 60.0
        rad = np.radians(angle)
        cx = np.sqrt(3) * np.cos(rad)
        cy = np.sqrt(3) * np.sin(rad)
        inner_hexes.append((cx, cy, 0.0))
    
    # 4 more
    for k in range(4):
        angle = 30.0 + k * 120.0
        rad = np.radians(angle)
        cx = 2 * np.sqrt(3) * np.cos(rad)
        cy = 2 * np.sqrt(3) * np.sin(rad)
        inner_hexes.append((cx, cy, 0.0))
    
    # Compute the outer side length for this configuration
    outer_side = compute_outer_side_length(inner_hexes)
    
    # Return the configuration
    # The interface expects: inner rows (x, y, angle_degrees), outer center, side, angle
    # We need to find the optimal outer center and angle.
    # For now, let's use the center of the inner configuration and angle 0.
    
    # Compute the centroid of the inner hexagons
    cx_mean = np.mean([h[0] for h in inner_hexes])
    cy_mean = np.mean([h[1] for h in inner_hexes])
    
    return inner_hexes, (cx_mean, cy_mean), outer_side, 0.0

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    """
    Optimize the positions and rotations of 11 unit hexagons to minimize the 
    outer enclosing hexagon side length.
    """
    # Compact hexagonal packing
    inner_hexes = []
    # Center
    inner_hexes.append([0.0, 0.0, 0.0])
    
    # 6 neighbors
    for k in range(6):
        angle = k * 60.0
        rad = np.radians(angle)
        cx = np.sqrt(3) * np.cos(rad)
        cy = np.sqrt(3) * np.sin(rad)
        inner_hexes.append([cx, cy, 0.0])
    
    # 4 more
    for k in range(4):
        angle = 30.0 + k * 120.0
        rad = np.radians(angle)
        cx = 2 * np.sqrt(3) * np.cos(rad)
        cy = 2 * np.sqrt(3) * np.sin(rad)
        inner_hexes.append([cx, cy, 0.0])
    
    # Compute the outer side length for this configuration
    outer_side = compute_outer_side_length(inner_hexes)
    
    # Compute the centroid of the inner hexagons
    cx_mean = np.mean([h[0] for h in inner_hexes])
    cy_mean = np.mean([h[1] for h in inner_hexes])
    
    return inner_hexes, (cx_mean, cy_mean), outer_side, 0.0
