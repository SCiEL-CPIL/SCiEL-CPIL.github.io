"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    deadline = time.time() + 140.0
    
    # Strategy: 
    # 1. Use a "flower" packing: 1 central hexagon, 6 surrounding it, 5 more in the next ring.
    # 2. Use flat-top orientation (angle=30 deg) to align with the outer hexagon (angle=0).
    # 3. Calculate the minimal outer side length that encloses this configuration.
    
    # Unit hexagon side length = 1.
    # Flat-top hexagon (angle=30) has vertices at angles 30, 90, 150, 210, 270, 330 relative to center.
    # Distance between centers of adjacent flat-top hexagons in a honeycomb lattice:
    # Horizontal neighbors: dx = 1.5, dy = sqrt(3)/2. Distance = sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3).
    # This matches the sum of inradii (sqrt(3)/2 + sqrt(3)/2 = sqrt(3)).
    
    # Define lattice vectors for flat-top hexagons
    # Vector to right neighbor: (1.5, sqrt(3)/2)
    # Vector to top-right neighbor: (0.75, sqrt(3)*1.5/2) = (0.75, 3*sqrt(3)/4)
    # Let's just generate the 12 positions manually or via simple logic.
    
    # Center at (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: We need 5 more. The "corners" of the hexagonal cluster of 7 are at distance 2*sqrt(3).
    # Let's list the 12 centers in the honeycomb lattice.
    # Axial coordinates (q, r) for hexagonal grid.
    # Flat-top mapping to Cartesian (x, y):
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    
    # We want a compact cluster of 12.
    # A hexagon of radius 2 has 1 + 6 + 12 = 19 cells. We only have 12.
    # A hexagon of radius 1 has 1 + 6 = 7 cells.
    # We need 5 more from the radius 2 ring.
    # The radius 2 ring has 12 cells. The "corners" of the radius 1 hexagon are at axial (2,0), (0,2), (-2,2), etc? No.
    # Let's just pick 5 cells from the second ring that are closest to the center to keep the bounding box small.
    # Actually, any 5 from the second ring will result in a similar bounding box if we center them.
    # To minimize the bounding hexagon, we should pick cells that are "balanced".
    # Let's take the 6 neighbors of the center, and 5 of the "corner" neighbors of the ring-1 hexagons.
    
    # Axial coordinates (q, r)
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 corners (neighbors of ring 1 corners):
    # Neighbor of (1,0) away from center: (2,0)
    # Neighbor of (1,-1) away: (2,-1)
    # Neighbor of (0,-1) away: (1,-2)
    # Neighbor of (-1,0) away: (-1,-1) -- wait, (-1,0) is ring 1. Neighbor away is (-2,0)? No.
    # Let's use cube coordinates or just generate neighbors.
    
    # Simpler: Just define the 12 points in Cartesian directly for a "flower" shape.
    # Center: (0,0)
    # 6 neighbors:
    # Angle 0: (sqrt(3), 0) -- wait, flat top means flat side up?
    # "Flat-top" usually means a horizontal edge on top.
    # Vertices of flat-top unit hexagon at angle 30:
    # (cos30, sin30), (cos90, sin90)...
    # (sqrt(3)/2, 1/2), (0, 1), (-sqrt(3)/2, 1/2), (-sqrt(3)/2, -1/2), (0, -1), (sqrt(3)/2, -1/2).
    # Inradius (distance center to edge midpoint) = cos(30)*1 = sqrt(3)/2.
    # Circumradius (distance center to vertex) = 1.
    
    # For two flat-top hexagons to be disjoint and touching, the distance between centers must be 2 * inradius = sqrt(3).
    # If we align them on a lattice:
    # Horizontal spacing: The width of a flat-top hexagon is 2 * inradius = sqrt(3).
    # So horizontal neighbors are at dx = sqrt(3), dy = 0?
    # Let's check. If I shift by (sqrt(3), 0), the right edge of left hex touches left edge of right hex.
    # Left hex right edge is at x = sqrt(3)/2. Right hex left edge is at x = sqrt(3) - sqrt(3)/2 = sqrt(3)/2. Yes.
    # So horizontal neighbors: (sqrt(3), 0).
    # Diagonal neighbors:
    # The top-right neighbor. The top edge is horizontal.
    # To go to the next row up, we shift by (sqrt(3)/2, 3/2)?
    # Let's check distance: sqrt( (sqrt(3)/2)^2 + (1.5)^2 ) = sqrt( 3/4 + 9/4 ) = sqrt(12/4) = sqrt(3). Correct.
    # So lattice vectors:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    # We want 12 points.
    # Center (0,0).
    # 6 neighbors:
    # v1, -v1, v2, -v2, v1+v2, -(v1+v2)?
    # Let's list 6 closest neighbors in this lattice.
    # 1. (sqrt(3), 0)
    # 2. (-sqrt(3), 0)
    # 3. (sqrt(3)/2, 1.5)
    # 4. (-sqrt(3)/2, -1.5)
    # 5. (sqrt(3)/2, -1.5)
    # 6. (-sqrt(3)/2, 1.5)
    # These are the 6 surrounding the center.
    
    # Now 5 more. We want to keep the cluster compact.
    # The "corners" of this 7-hex cluster are at:
    # (sqrt(3), 0) + (sqrt(3)/2, 1.5) = (3*sqrt(3)/2, 1.5)
    # (sqrt(3)/2, 1.5) + (-sqrt(3)/2, 1.5) = (0, 3)
    # (-sqrt(3)/2, 1.5) + (-sqrt(3), 0) = (-3*sqrt(3)/2, 1.5)
    # (-sqrt(3), 0) + (-sqrt(3)/2, -1.5) = (-3*sqrt(3)/2, -1.5)
    # (-sqrt(3)/2, -1.5) + (sqrt(3)/2, -1.5) = (0, -3)
    # (sqrt(3)/2, -1.5) + (sqrt(3), 0) = (3*sqrt(3)/2, -1.5)
    
    # These 6 points form a larger hexagon. We need to pick 5 of them.
    # Picking any 5 will break symmetry slightly, but the bounding box will be determined by the 5 furthest points.
    # To minimize the max distance to the center (which roughly correlates with outer side length),
    # we should pick 5 that are as "central" as possible? No, they are all at the same distance from center:
    # Distance of (3*sqrt(3)/2, 1.5) from origin:
    # sqrt( 27/4 + 9/4 ) = sqrt(36/4) = 3.
    # All 6 are at distance 3.
    # If we pick 5, the bounding hexagon will be slightly smaller than if we picked 6?
    # Actually, the outer hexagon must contain all vertices of all inner hexagons.
    # The vertices of the inner hexagons extend 1 unit from their centers.
    # So the "effective" radius of the cluster is 3 + 1 = 4?
    # Wait, the outer hexagon side length S relates to its inradius R_in = S * sqrt(3)/2.
    # The outer hexagon must contain the union of the 12 inner hexagons.
    # The farthest point from the center (0,0) in the cluster of centers is at distance 3.
    # The vertices of the outermost inner hexagons extend to distance 3 + 1 = 4 from the origin?
    # Not necessarily. The vertex might point towards the center.
    # However, to be safe, the inradius of the outer hexagon must be at least the max distance from the origin to any point in the inner hexagons.
    # Max distance from origin to any point in an inner hexagon centered at C is |C| + 1 (if the vertex points away).
    # So we need R_in_outer >= 4.
    # S * sqrt(3)/2 >= 4 => S >= 8 / sqrt(3) ≈ 4.618.
    
    # Let's verify if S=4.618 is sufficient.
    # The outer hexagon has inradius 4.
    # The inner hexagons have circumradius 1.
    # The centers are at distance <= 3.
    # Is it possible that a vertex of an inner hexagon sticks out of the outer hexagon?
    # The outer hexagon is defined by 6 lines.
    # If we align the outer hexagon with the same orientation as the inner ones (angle 30),
    # the "flat" sides are parallel.
    # The distance from the center to the "flat" side of an inner hexagon is its inradius = sqrt(3)/2 ≈ 0.866.
    # The distance from the center to the "vertex" is 1.
    
    # Let's calculate the required outer side length more precisely.
    # We have 12 centers.
    # We place them at the 7 central points and 5 of the 6 ring-2 points.
    # Let's pick the 5 ring-2 points that are NOT the "top" one (0,3) to see if we can save space?
    # Actually, due to symmetry, any 5 will result in a similar bounding box.
    # Let's just compute the bounding box of the 12 hexagons.
    
    # Generate the 12 centers
    centers = [
        (0.0, 0.0),
        (math.sqrt(3), 0.0),
        (-math.sqrt(3), 0.0),
        (math.sqrt(3)/2, 1.5),
        (-math.sqrt(3)/2, -1.5),
        (math.sqrt(3)/2, -1.5),
        (-math.sqrt(3)/2, 1.5),
        # Ring 2 (pick 5)
        (3*math.sqrt(3)/2, 1.5),
        (-3*math.sqrt(3)/2, 1.5),
        (-3*math.sqrt(3)/2, -1.5),
        (0.0, -3.0),
        (3*math.sqrt(3)/2, -1.5)
    ]
    
    # Angle of inner hexagons: 30 degrees (flat-top)
    angle_deg = 30.0
    
    # Construct inner list: [x, y, angle]
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, angle_deg])
        
    # Now determine the minimal outer hexagon side length.
    # The outer hexagon is centered at (0,0) with angle 30 degrees (aligned with inner).
    # We need to find the smallest S such that all inner hexagons are inside.
    # This is equivalent to finding the maximum "distance" from the origin in the 6 directions of the hexagon normals,
    # considering the extent of the inner hexagons in those directions.
    
    # The 6 normal directions for a flat-top hexagon (angle 30) are:
    # The flat sides are at angles 90, 150, 210, 270, 330, 30?
    # Vertices are at 30, 90, 150, 210, 270, 330.
    # Flat sides are perpendicular to the vectors pointing to the midpoints of the sides.
    # Midpoint of side between 30 and 90 is at 60.
    # So normals are at 60, 120, 180, 240, 300, 0?
    # Let's check.
    # Side 1: vertices at 30 and 90. Midpoint angle 60. Normal angle 60.
    # Side 2: vertices at 90 and 150. Midpoint angle 120. Normal angle 120.
    # Side 3: vertices at 150 and 210. Midpoint angle 180. Normal angle 180.
    # Side 4: vertices at 210 and 270. Midpoint angle 240. Normal angle 240.
    # Side 5: vertices at 270 and 330. Midpoint angle 300. Normal angle 300.
    # Side 6: vertices at 330 and 30. Midpoint angle 0 (or 360). Normal angle 0.
    
    normals = [math.radians(a) for a in [0, 60, 120, 180, 240, 300]]
    
    max_proj = 0.0
    for cx, cy in centers:
        for theta in normals:
            # Project the center onto the normal
            proj_center = cx * math.cos(theta) + cy * math.sin(theta)
            
            # Project the extent of the inner hexagon in this normal direction.
            # The max projection of a unit flat-top hexagon (angle 30) onto a normal at angle theta.
            # The support function of a regular polygon.
            # For a hexagon with circumradius R=1, the support function h(theta) is:
            # h(theta) = R * cos(pi/6 - delta) where delta is the angle between theta and the nearest vertex normal?
            # Simpler: Just calculate the max projection of the 6 vertices of the inner hexagon.
            
            # Vertices of inner hexagon centered at (cx, cy) with angle 30:
            # v_i = (cx + cos(30 + 60i), cy + sin(30 + 60i))
            # We want max over i of (v_i.x * cos(theta) + v_i.y * sin(theta))
            # = cx*cos(theta) + cy*sin(theta) + max_i (cos(30+60i)*cos(theta) + sin(30+60i)*sin(theta))
            # = proj_center + max_i (cos(theta - (30+60i)))
            
            max_vertex_proj = 0.0
            for i in range(6):
                vertex_angle = math.radians(30 + 60 * i)
                proj_v = math.cos(theta - vertex_angle)
                if proj_v > max_vertex_proj:
                    max_vertex_proj = proj_v
            
            total_proj = proj_center + max_vertex_proj
            if total_proj > max_proj:
                max_proj = total_proj
                
    # The inradius of the outer hexagon must be at least max_proj.
    # R_in_outer = S * sqrt(3) / 2
    # S = 2 * max_proj / sqrt(3)
    
    outer_side = 2.0 * max_proj / math.sqrt(3)
    
    # Add a tiny epsilon to ensure strict containment if the evaluator is strict
    outer_side += 1e-9
    
    return inner, (0.0, 0.0), outer_side, 30.0
# EVOLVE-BLOCK-END
