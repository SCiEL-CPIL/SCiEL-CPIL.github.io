import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQ3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Return the 6 vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return verts

def _hex_in_hexagon(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if a unit hexagon at (cx,cy) with angle_deg fits inside outer hexagon
    (center outer_cx,outer_cy; side outer_side; rotation outer_angle_deg).
    Returns True if all 6 vertices of inner hex are inside outer hex."""
    # Outer hexagon apothem
    outer_apothem = outer_side * APOTHEM
    outer_a = math.radians(outer_angle_deg)
    
    # 6 inward-pointing normals of the outer hexagon
    for k in range(6):
        # The outward normal of face k
        theta = outer_a + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        # Distance from outer center to face along normal = apothem
        # For point to be inside: dot((px-ocx, py-ocy), (nx,ny)) <= apothem
        verts = _hex_vertices(cx, cy, angle_deg)
        for vx, vy in verts:
            dist = (vx - outer_cx) * nx + (vy - outer_cy) * ny
            if dist > outer_apothem + 1e-9:
                return False
    return True

def _hexes_disjoint(pts, angles):
    """Check if N unit hexagons are pairwise disjoint (non-overlapping).
    pts: list of (cx,cy), angles: list of angle_degrees.
    Two hexagons overlap iff any edge of one intersects any edge of the other,
    or one is contained in the other. We use a vertex-edge separation test.
    For regular hexagons of same size, disjointness means the minimum distance
    between any two hexagons is >= 0. We use: for each pair, check that no
    vertex of hex A is inside hex B AND no vertex of hex B is inside hex A,
    AND no edges cross. For same-size convex polygons, if no vertex is inside
    the other and no edges cross, they're disjoint or touching.
    A simpler sufficient condition: the distance between the two centers must
    be >= 2*APOTHEM when aligned, but this is not sufficient for arbitrary rotations.
    
    Actually for two unit regular hexagons, the minimum center-to-center distance
    for non-overlap depends on relative rotation. The worst case (aligned) requires
    center distance >= 2*APOTHEM = sqrt(3). For 30-degree rotation, it can be less.
    
    Let's use a robust check: for each pair, check if any vertex of one is inside
    the other, and check edge-edge intersections.
    """
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            verts_i = _hex_vertices(pts[i][0], pts[i][1], angles[i])
            verts_j = _hex_vertices(pts[j][0], pts[j][1], angles[j])
            
            # Check if any vertex of i is inside j
            inside = False
            for vx, vy in verts_i:
                if _point_in_hexagon(vx, vy, verts_j):
                    inside = True
                    break
            if inside:
                return False
            
            # Check if any vertex of j is inside i
            for vx, vy in verts_j:
                if _point_in_hexagon(vx, vy, verts_i):
                    return False
            
            # Check edge-edge intersections
            for e1 in range(6):
                for e2 in range(6):
                    if _segments_intersect(verts_i[e1], verts_i[(e1+1)%6],
                                           verts_j[e2], verts_j[(e2+1)%6]):
                        return False
    return True

def _point_in_hexagon(px, py, verts):
    """Check if point (px,py) is inside the hexagon defined by verts (list of 6 points)."""
    # Ray casting method
    inside = False
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1)%n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _outer_side_from_points(pts, angles, outer_cx, outer_cy, outer_angle_deg):
    """Given inner hexagon centers/angles, find the minimum outer side length
    such that all inner hexes fit inside the outer hexagon centered at (outer_cx, outer_cy)
    with rotation outer_angle_deg. Returns the minimum side length."""
    # For each inner hexagon vertex, compute the maximum projection onto each of the
    # 6 outer normals. The required apothem is the max of these projections.
    outer_a = math.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        theta = outer_a + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    for cx, cy in pts:
        ang = angles[cx] if isinstance(angles, dict) else angles[list(pts).index((cx,cy))] if (cx,cy) in pts else 0
        verts = _hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            for nx, ny in normals:
                proj = (vx - outer_cx) * nx + (vy - outer_cy) * ny
                if proj > max_proj:
                    max_proj = proj
    
    # Required apothem = max_proj, side = apothem / APOTHEM
    return max_proj / APOTHEM

def construct():
    start_time = time.time()
    time_limit = 110.0  # Stay well within 120s
    
    # Strategy: Use differential evolution to optimize positions and rotations
    # of N hexagons, minimizing the bounding outer hexagon side length.
    # 
    # Parameters: for each hex i: x_i, y_i, angle_i
    # Total: 3*N parameters
    #
    # Objective: minimize the side length of the smallest enclosing regular hexagon
    # that contains all inner hexagons.
    #
    # For efficiency, we fix the outer hexagon's orientation to 0 degrees and
    # center it at the centroid of the inner hexagons. This is WLOG since we
    # can always rotate/translate the whole configuration.
    
    n = N  # 11
    
    # Initial guess: hexagonal packing
    # In a hexagonal grid, adjacent hexagons have center distance = 2*APOTHEM = sqrt(3)
    d = SQ3  # center-to-center distance for touching hexagons
    
    # Arrange 11 hexagons in a compact cluster
    # Center + 6 around + 4 more
    axial_positions = [
        (0, 0),       # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (-2, 0), (0, 2), (1, -2)  # ring 2 partial
    ]
    
    # Convert axial to cartesian
    def axial_to_cart(q, r):
        x = SQ3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    init_pts = [axial_to_cart(q, r) for q, r in axial_positions[:n]]
    init_angles = [0.0] * n
    
    # Compute initial objective
    def objective(params):
        """params: array of length 3*N: [x1, y1, a1, x2, y2, a2, ...]"""
        if time.time() - start_time > time_limit:
            return 1e6
        
        pts = []
        angles = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            pts.append((x, y))
            angles.append(a)
        
        # Check disjointness - if overlapping, return large penalty
        if not _hexes_disjoint(pts, angles):
            # Compute how much overlap to use as penalty
            # Simple: return a large value
            return 10.0 + n * 0.1
        
        # Compute enclosing hexagon side length
        # Center the configuration at origin for the outer hexagon
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        
        outer_side = _outer_side_from_points(pts, angles, cx, cy, 0.0)
        
        return outer_side
    
    # Use differential evolution for global optimization
    # Bounds: positions in [-10, 10], angles in [0, 60) degrees
    bounds = []
    for i in range(n):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 60))    # angle in degrees
    
    # Initial population
    init_x = []
    for i in range(n):
        init_x.append(init_pts[i][0])
        init_x.append(init_pts[i][1])
        init_x.append(init_angles[i])
    
    # Run differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=42,
            maxiter=200,
            tol=1e-6,
            polish=True,
            update_interval=10,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.7
        )
        
        best_params = result.x
        best_side = result.fun
        
        if time.time() - start_time < time_limit - 5:
            # Polish with Nelder-Mead
            def polish_obj(params):
                if time.time() - start_time > time_limit:
                    return 1e6
                return objective(params)
            
            for _ in range(3):
                res = minimize(polish_obj, best_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < best_side:
                    best_params = res.x
                    best_side = res.fun
                if time.time() - start_time > time_limit - 3:
                    break
        
    except Exception:
        # Fallback to initial configuration
        best_params = np.array(init_x)
        best_side = objective(best_params)
    
    # Extract final configuration
    pts = []
    angles = []
    for i in range(n):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        a = float(best_params[3*i+2]) % 60.0  # Normalize angle
        pts.append((x, y))
        angles.append(a)
    
    # Recompute the actual enclosing side length
    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    outer_side = _outer_side_from_points(pts, angles, cx, cy, 0.0)
    
    # Small safety margin
    outer_side *= (1.0 + 1e-9)
    
    inner = [[float(x), float(y), float(a)] for (x, y), a in zip(pts, angles)]
    center = [float(cx), float(cy)]
    side = float(outer_side)
    rotation = 0.0
    
    return inner, center, side, rotation
