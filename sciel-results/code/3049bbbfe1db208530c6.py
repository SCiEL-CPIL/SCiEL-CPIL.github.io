"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    # Vertices of a regular hexagon (flat-top orientation at angle=0)
    # For a regular hexagon with side length s, the distance from center to vertex is s
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees relative to the hexagon's orientation
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    x = cx + s * np.cos(angles)
    y = cy + s * np.sin(angles)
    return np.column_stack([x, y])

def _point_in_polygon(p, poly):
    """Ray casting to check if point p is inside polygon poly (n x 2)."""
    x, y = p
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _polygons_overlap(poly1, poly2, eps=1e-9):
    """Check if two convex polygons overlap. Returns True if they overlap."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for p in poly1:
        if _point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if _point_in_polygon(p, poly1):
            return True
    # Check if any edges intersect
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        for j in range(n2):
            if _segments_intersect(poly1[i], poly1[(i+1) % n1], poly2[j], poly2[(j+1) % n2]):
                return True
    return False

def _hexagon_distance_to_container(hex_verts, outer_verts):
    """Check if hexagon is fully inside outer hexagon. Returns True if contained."""
    # All vertices of hex must be inside outer
    for p in hex_verts:
        if not _point_in_polygon(p, outer_verts):
            return False
    return True

def _min_outer_side(inner_positions, outer_center, outer_angle):
    """
    Find the minimum side length of an outer regular hexagon with given center and angle
    that contains all inner hexagons.
    inner_positions: list of (x, y, angle_deg)
    Returns the minimum side length, or inf if impossible.
    """
    # The outer hexagon's side length must be large enough to contain all vertices of all inner hexagons
    # For a regular hexagon with center C, side s, and rotation angle, a point P is inside if
    # its distance in each of the 6 normal directions is <= s * sqrt(3) / 2 (the apothem)
    
    # Actually, for a regular hexagon with side length s, the apothem (distance from center to midpoint of side) is s * sqrt(3) / 2
    # A point is inside the hexagon if for each of the 6 face normals, the projection is <= apothem
    
    # The 6 face normals for a hexagon with rotation angle theta:
    # Faces are at angles theta + 30, theta + 90, theta + 150, theta + 210, theta + 270, theta + 330 degrees
    # Wait, let me think more carefully.
    
    # A regular hexagon with side s, centered at origin, with one vertex at angle 0:
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # The faces (edges) are between consecutive vertices
    # The outward normals to the faces point at angles 30, 90, 150, 210, 270, 330
    # The apothem is s * sqrt(3) / 2
    
    # So for a hexagon with center C, side s, rotation angle alpha:
    # A point P is inside if for each k in 0..5:
    #   dot(P - C, normal_k) <= s * sqrt(3) / 2
    # where normal_k points at angle alpha + 30 + k * 60 degrees
    
    theta = np.radians(outer_angle)
    normals = []
    for k in range(6):
        ang = theta + np.radians(30 + k * 60)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for (x, y, ang) in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_points.extend(verts)
    all_points = np.array(all_points)
    
    # For each face normal, compute the maximum projection of any point
    # The required apothem is max over all points and all normals of dot(P - C, normal)
    # Then s = apothem / (sqrt(3)/2)
    
    C = np.array(outer_center)
    rel = all_points - C
    # rel: (M, 2), normals: (6, 2)
    # projections: (M, 6)
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)  # (6,)
    required_apothem = np.max(max_proj)
    
    if required_apothem <= 0:
        return 0.0  # All points at or inside center (degenerate)
    
    s = required_apothem / (np.sqrt(3) / 2)
    return s

def _is_valid_configuration(inner_positions, outer_center, outer_side, outer_angle, eps=1e-6):
    """Check if the configuration is valid: no overlaps, all inside container."""
    # Check pairwise overlaps of inner hexagons
    n = len(inner_positions)
    verts_list = []
    for (x, y, ang) in inner_positions:
        verts_list.append(_hex_vertices(x, y, 1.0, ang))
    
    for i in range(n):
        for j in range(i + 1, n):
            if _polygons_overlap(verts_list[i], verts_list[j], eps):
                return False
    
    # Check all inner hexagons are inside outer
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(n):
        for p in verts_list[i]:
            if not _point_in_polygon(p, outer_verts):
                return False
    
    return True

def construct():
    """
    Construct a packing of 11 unit regular hexagons inside a regular hexagon.
    Strategy: Use a hexagonal lattice arrangement of the 11 inner hexagons as a rigid cluster.
    Optimize the cluster's rotation and the outer hexagon's orientation to minimize the outer side.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Define the hexagonal lattice positions for 11 unit hexagons
    # In a hexagonal lattice with unit hexagons (side 1), the distance between centers
    # of adjacent hexagons is 2 * (sqrt(3)/2) = sqrt(3)
    # Wait, for unit hexagons (side 1), the distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Actually, let me reconsider. Two unit hexagons (side 1) that are edge-to-edge:
    # The distance between their centers is 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732
    
    # Hexagonal lattice: positions in axial coordinates (q, r)
    # For a "flower" pattern with 1 center + 6 around + 4 more, or a compact arrangement
    
    # Let's use a hexagonal arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # Total: 11
    
    # Or better: a compact hexagonal cluster
    # Center: (0,0)
    # Ring 1: 6 hexagons at distance sqrt(3) in 6 directions
    # Ring 2: 4 more hexagons to make 11 total
    
    # Let's use axial coordinates for a hex grid
    # Axial to pixel: x = sqrt(3) * (q + r/2), y = 3/2 * r
    
    # A compact arrangement of 11:
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) - that's 7 (center + ring 1)
    # Add 4 more: (2,0), (1,1), (-2,1), (-1,-1) or similar
    
    # Let me just define specific positions that form a compact cluster
    # Using the fact that adjacent hex centers are sqrt(3) apart
    
    d = np.sqrt(3)  # distance between adjacent hex centers
    
    # Compact cluster: hexagonal shape with 3 in a row pattern
    # Let's use:
    # (0, 0)
    # (d, 0), (-d, 0)
    # (d/2, d*sqrt(3)/2), (d/2, -d*sqrt(3)/2)
    # (-d/2, d*sqrt(3)/2), (-d/2, -d*sqrt(3)/2)
    # That's 7. Add 4 more:
    # (2d, 0), (3d/2, d*sqrt(3)/2), (3d/2, -d*sqrt(3)/2), (d, d*sqrt(3))
    # Hmm, let me just use a simple hexagonal lattice pattern
    
    # Simple approach: use axial coordinates
    # positions in axial coords (q, r) -> pixel: (sqrt(3)*(q + r/2), 1.5*r)
    # Scale by 1 (since we want unit hexagons, the lattice spacing is already set)
    
    # Actually, for unit hexagons, the lattice vectors are:
    # a1 = (sqrt(3), 0)
    # a2 = (sqrt(3)/2, 3/2)
    # So position = q*a1 + r*a2 = (sqrt(3)*q + sqrt(3)/2*r, 3/2*r)
    
    # Let's define 11 positions in a compact hexagonal arrangement
    # Center (0,0)
    # 6 neighbors: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more: (2,0), (1,1), (-2,1), (-1,-1)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-2, 1), (-1, -1)
    ]
    
    # Convert to pixel coordinates
    lattice_positions = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        lattice_positions.append((x, y))
    
    # Center the cluster
    xs = [p[0] for p in lattice_positions]
    ys = [p[1] for p in lattice_positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    lattice_positions = [(x - cx, y - cy) for x, y in lattice_positions]
    
    def evaluate(cluster_rotation, outer_rotation, outer_center_offset=(0, 0)):
        """Evaluate the outer side length for given cluster rotation and outer rotation."""
        # Rotate all inner hexagon positions by cluster_rotation
        cr = np.radians(cluster_rotation)
        inner_positions = []
        for (x, y) in lattice_positions:
            xr = x * np.cos(cr) - y * np.sin(cr)
            yr = x * np.sin(cr) + y * np.cos(cr)
            inner_positions.append((xr, yr, cluster_rotation))
        
        # The outer hexagon center is at outer_center_offset
        oc = outer_center_offset
        
        # Compute minimum outer side
        side = _min_outer_side(inner_positions, oc, outer_rotation)
        return side
    
    # Optimize over cluster_rotation and outer_rotation
    # Both are in degrees, periodic with period 60
    
    best_side = float('inf')
    best_params = None
    
    # Grid search first
    n_angles = 30
    for i in range(n_angles):
        if time.time() > deadline - 30:
            break
        cr = i * 60.0 / n_angles
        for j in range(n_angles):
            if time.time() > deadline - 30:
                break
            orr = j * 60.0 / n_angles
            side = evaluate(cr, orr)
            if side < best_side:
                best_side = side
                best_params = (cr, orr)
    
    # Local optimization
    if best_params is not None and time.time() < deadline - 30:
        def objective(params):
            cr, orr = params
            return evaluate(cr, orr)
        
        x0 = np.array(best_params)
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
        
        if res.fun < best_side:
            best_side = res.fun
            best_params = (res.x[0], res.x[1])
    
    # Additional local optimization from several starting points
    if time.time() < deadline - 20:
        for start_cr in [0, 15, 30, 45]:
            if time.time() > deadline - 20:
                break
            for start_orr in [0, 15, 30, 45]:
                if time.time() > deadline - 20:
                    break
                
                def objective2(params):
                    cr, orr = params
                    return evaluate(cr, orr)
                
                x0 = np.array([start_cr, start_orr])
                res = minimize(objective2, x0, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
                
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = (res.x[0], res.x[1])
    
    # Construct final answer
    if best_params is None:
        # Fallback
        cr, orr = 0.0, 0.0
    else:
        cr, orr = best_params
    
    # Final evaluation
    cr_final = cr
    orr_final = orr
    
    # Rotate inner positions
    cr_rad = np.radians(cr_final)
    inner = []
    for (x, y) in lattice_positions:
        xr = x * np.cos(cr_rad) - y * np.sin(cr_rad)
        yr = x * np.sin(cr_rad) + y * np.cos(cr_rad)
        inner.append([xr, yr, cr_final])
    
    outer_center = (0.0, 0.0)
    outer_side = _min_outer_side(inner, outer_center, orr_final)
    
    return inner, outer_center, outer_side, orr_final
# EVOLVE-BLOCK-END
