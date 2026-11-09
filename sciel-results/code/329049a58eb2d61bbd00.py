# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_min_enclosing_hexagon(points, angle_deg):
    """
    Computes the minimum side length of a regular hexagon with given angle 
    and center at origin that encloses all points.
    """
    # Hexagon vertices are at radius d, angles: angle, angle+60, ..., angle+300
    # Edges are perpendicular to normals at angles: angle+30, angle+90, ..., angle+330
    # The support function of a regular hexagon with side d and angle a in direction u (unit)
    # is d * |cos(u_angle - a)| / cos(30) is not quite right for all directions.
    
    # Actually, a regular hexagon with side d and angle a (first vertex at a) has:
    # Distance from center to edge (apothem) = d * sqrt(3) / 2
    # The projection of any point (x,y) onto the normal of an edge must be <= apothem.
    
    # Let's just compute the max projection onto the 6 edge normals.
    # Edge normals for a hexagon with vertices at a, a+60...
    # If vertex 0 is at angle a, edge 0 is between v5 and v0? No, standard is v0 to v1.
    # Let's use the property that the distance from center to the line segment connecting 
    # vertices at angle a-30 and a+30 is d*sqrt(3)/2.
    # The normal to this edge is at angle a.
    
    # So the 6 normals are at angles: a, a+60, ..., a+300.
    # Wait, if vertices are at a, a+60...
    # The edge between v0(a) and v1(a+60) has midpoint at angle a+30.
    # The normal to this edge is at angle a+30.
    # So the 6 normals are at a+30, a+90, ..., a+330.
    
    apothem_ratio = np.sqrt(3) / 2.0
    # d = max_projection / apothem_ratio
    
    max_proj = 0.0
    for i in range(6):
        normal_angle = np.radians(angle_deg + 30 + i * 60)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        # Project points onto normal
        # We need max of (p.x * nx + p.y * ny) and min?
        # Since the hexagon is centered at origin, it covers -d*apo to +d*apo in each normal direction.
        # So we need max |projection|
        projs = points[:, 0] * nx + points[:, 1] * ny
        current_max = np.max(np.abs(projs))
        if current_max > max_proj:
            max_proj = current_max
            
    return max_proj / apothem_ratio

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Define the 12 inner hexagons in a "flower" arrangement
    # Center hexagon at (0,0)
    # 6 surrounding hexagons touching the center
    # 3 more hexagons filling the gaps between the surrounding ones (at distance sqrt(7) from center? No, distance from center to gap vertex is different)
    
    # Let's generate coordinates.
    # Unit hexagon side = 1.
    # Flat-top hexagon vertices: (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Centers of touching hexagons:
    # Neighbors of (0,0) are at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300.
    # Wait, distance between centers of touching flat-top hexagons?
    # If they touch at an edge, the distance is 2 * apothem = sqrt(3).
    # Directions of edges: 0, 60, 120... (normals).
    # So centers are at (sqrt(3), 0), (sqrt(3)/2, 3/2), etc.
    # Let h = sqrt(3)/2.
    # Centers:
    # (0,0)
    # (sqrt(3), 0) -> (1.732, 0)
    # (sqrt(3)/2, 1.5) -> (0.866, 1.5)
    # (-sqrt(3)/2, 1.5)
    # (-sqrt(3), 0)
    # (-sqrt(3)/2, -1.5)
    # (sqrt(3)/2, -1.5)
    
    # Now add 3 more.
    # The "gaps" are at distance ?
    # Consider the hexagon at (sqrt(3), 0). Its right-most vertex is at (sqrt(3)+1, 0).
    # Consider the hexagon at (sqrt(3)/2, 1.5). Its top-most vertex?
    # The gap is between (sqrt(3),0), (sqrt(3)/2, 1.5), and (-sqrt(3)/2, 1.5)? No, those don't touch each other?
    # Distance between (sqrt(3),0) and (sqrt(3)/2, 1.5):
    # dx = sqrt(3)/2, dy = 1.5. dist^2 = 3/4 + 9/4 = 12/4 = 3. dist = sqrt(3).
    # So they DO touch.
    # So the 6 surrounding hexagons form a closed ring?
    # Yes, 6 hexagons around 1 central hexagon forms a ring of 6, total 7.
    # The "gaps" are the 3 concave corners of this ring of 6.
    # Let's find the position of a new hexagon that touches 3 existing ones.
    # It would touch (sqrt(3),0), (sqrt(3)/2, 1.5), and (0,0)? No, (0,0) is already surrounded.
    # It must touch (sqrt(3),0) and (sqrt(3)/2, 1.5).
    # And the third neighbor is in the gap.
    # The vector from (0,0) to the new center?
    # Let the new center be C.
    # dist(C, (sqrt(3),0)) = sqrt(3)
    # dist(C, (sqrt(3)/2, 1.5)) = sqrt(3)
    # This implies C is at distance sqrt(3) from both.
    # The bisector of the segment connecting the two neighbors?
    # Midpoint M = ( (sqrt(3) + sqrt(3)/2)/2, (0+1.5)/2 ) = (3sqrt(3)/4, 0.75).
    # Direction of segment: (sqrt(3)/2, 1.5).
    # Perpendicular: (-1.5, sqrt(3)/2). Normalized?
    # Let's just use symmetry. The 3 extra hexagons are at distance ? from origin.
    # In the standard hexagonal lattice, if we have a flower of 7, the next layer starts at distance 3*sqrt(3)?
    # No, the "gap" filling hexagons are part of the second layer.
    # The distance from center to the center of a gap-filling hexagon is 3?
    # Let's check distance from (0,0) to (1.5, 0)?
    # If we place a hex at (1.5, 0), does it touch (sqrt(3), 0)?
    # dist = |1.5 - 1.732| = 0.232. Too close, overlap.
    
    # Let's calculate the position correctly.
    # The 6 surrounding hexagons have centers at radius R1 = sqrt(3).
    # Angles 0, 60, ...
    # The gap is at angle 30, 150, 270.
    # Let's assume the new center is at radius R2, angle 30.
    # It must touch the hexagon at angle 0 (center P1) and angle 60 (center P2).
    # P1 = (sqrt(3), 0). P2 = (sqrt(3)/2, 1.5).
    # New Center P3 = (R2 cos 30, R2 sin 30) = (R2 * sqrt(3)/2, R2 / 2).
    # dist(P3, P1)^2 = 3
    # (R2 sqrt(3)/2 - sqrt(3))^2 + (R2/2 - 0)^2 = 3
    # 3/4 R2^2 - 2 * R2 sqrt(3)/2 * sqrt(3) + 3 + R2^2 / 4 = 3
    # R2^2 - 3 R2 + 3 = 3
    # R2^2 - 3 R2 = 0
    # R2 = 3 (since R2 > 0).
    
    # So the 3 extra hexagons are at distance 3 from the origin, at angles 30, 150, 270.
    # Let's verify dist(P3, P2).
    # P3 = (3 sqrt(3)/2, 1.5). P2 = (sqrt(3)/2, 1.5).
    # dx = 3 sqrt(3)/2 - sqrt(3)/2 = sqrt(3). dy = 0.
    # dist = sqrt(3). Correct.
    
    # So the 12 centers are:
    # 1 at (0,0)
    # 6 at radius sqrt(3), angles 0, 60, 120, 180, 240, 300
    # 3 at radius 3, angles 30, 150, 270
    
    centers = []
    centers.append((0, 0))
    for i in range(6):
        angle = i * 60
        rad = np.radians(angle)
        centers.append((np.sqrt(3) * np.cos(rad), np.sqrt(3) * np.sin(rad)))
    for i in range(3):
        angle = 30 + i * 120
        rad = np.radians(angle)
        centers.append((3 * np.cos(rad), 3 * np.sin(rad)))
        
    # Create inner hex data
    # All inner hexagons have angle 0 (flat-top)
    inner_hex_data = np.zeros((12, 3))
    for i, (cx, cy) in enumerate(centers):
        inner_hex_data[i] = [cx, cy, 0]
        
    # Generate all vertices of the 12 hexagons
    # Flat-top unit hexagon vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    verts_local = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    all_verts = []
    for cx, cy in centers:
        for v in verts_local:
            all_verts.append([cx + v[0], cy + v[1]])
            
    all_verts = np.array(all_verts)
    
    # Try outer hexagon angles 0 and 30
    best_S = float('inf')
    best_angle = 0
    
    for ang in [0, 30]:
        S = _compute_min_enclosing_hexagon(all_verts, ang)
        if S < best_S:
            best_S = S
            best_angle = ang
            
    # The outer hexagon is centered at origin
    outer_hex_data = np.array([0, 0, best_angle])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length

# EVOLVE-BLOCK-END
