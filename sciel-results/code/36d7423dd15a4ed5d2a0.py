"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin before 120s limit
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal cluster arrangement for 11 unit hexagons
    # Using a compact hexagonal pattern
    # Flat-top hexagons with side length 1
    # Adjacent hex centers are distance 2 apart horizontally, sqrt(3) vertically
    
    # Place hexagons in a hexagonal cluster:
    # Row structure (using offset coordinates):
    # We want 11 hexagons in the most compact shape
    
    # Let's use axial coordinates and convert to pixel
    # For flat-top hexagons:
    # x = 3/2 * q
    # y = sqrt(3)/2 * q + sqrt(3) * r
    # But for unit hexagons (side=1), the spacing between adjacent centers is 2
    
    # Actually, for regular hexagons with side length 1:
    # The distance between centers of adjacent hexagons (sharing an edge) is 2
    # (since each has inradius sqrt(3)/2, so 2 * sqrt(3)/2 = sqrt(3)... no)
    
    # Wait: inradius (apothem) of unit hexagon = sqrt(3)/2
    # When two hexagons share an edge, the distance between centers = 2 * apothem = sqrt(3)
    # No, that's wrong. Let me recalculate.
    
    # A regular hexagon with side length s has:
    # - Circumradius (center to vertex) = s
    # - Inradius (center to edge midpoint) = s * sqrt(3) / 2
    
    # For s=1: circumradius=1, inradius=sqrt(3)/2
    
    # Two hexagons sharing an edge: the edge is at distance inradius from each center,
    # on opposite sides. So center-to-center distance = 2 * inradius = sqrt(3)
    
    # So adjacent hex centers are sqrt(3) apart, not 2!
    
    # Let me redo with correct spacing
    # For flat-top hexagons with side 1:
    # Horizontal neighbors: distance = 2 * inradius = sqrt(3)? No...
    
    # Actually, I need to be more careful. Let me use the standard hex grid formula.
    # For a hex grid with hexagon side length 1 (flat-top):
    # - Width of hexagon = 2 (point to point horizontally)
    # - Height of hexagon = sqrt(3) (flat to flat vertically)
    # - Horizontal spacing between adjacent columns = 3/2 * width/2 = 3/2? 
    
    # Let me just use the fact that in a hex grid, the 6 neighbors of a center hex
    # are at distance = 2 * inradius = sqrt(3) from the center, at angles 0, 60, 120, 180, 240, 300
    
    # Wait, no. The neighbors share edges, so the center-to-center distance is 2 * inradius = sqrt(3)
    # But the direction depends on the hex orientation.
    
    # For flat-top hexagons, the 6 edge midpoints are at angles 0, 60, 120, 180, 240, 300 from center
    # at distance inradius = sqrt(3)/2
    # So neighbor centers are at distance 2*inradius = sqrt(3) in those same directions
    
    hex_positions = []
    
    # Center
    hex_positions.append((0.0, 0.0))
    
    # 6 neighbors at distance sqrt(3) in directions 0, 60, ..., 300
    for i in range(6):
        angle = np.radians(60 * i)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        hex_positions.append((x, y))
    
    # 4 more in the second ring
    # Second ring neighbors are at distance 2*sqrt(3) or sqrt(3)*sqrt(3)=3 from center
    # In a hex grid, the second ring has 12 positions
    # Let me add 4 more that are compact
    
    # Second ring positions (axial coordinates at distance 2 from center)
    # Using cube coordinates: (q, r, s) with q+r+s=0, distance = max(|q|,|r|,|s|)
    # For distance 2: various combinations
    
    # Let me just add positions that are adjacent to existing ones
    # Position (sqrt(3), 0) has neighbors including (2*sqrt(3), 0) which is distance 2*sqrt(3) from origin
    
    second_ring_candidates = []
    for i in range(6):
        angle = np.radians(60 * i)
        # Distance 2*sqrt(3) in same direction
        second_ring_candidates.append((2*sqrt3 * np.cos(angle), 2*sqrt3 * np.sin(angle)))
        # Also positions between the main directions
        angle2 = np.radians(60 * i + 30)
        second_ring_candidates.append((3.0 * np.cos(angle2), 3.0 * np.sin(angle2)))
    
    # Add 4 from second ring
    for pos in second_ring_candidates:
        if len(hex_positions) >= 11:
            break
        # Check if this position is close to existing cluster
        hex_positions.append(pos)
    
    # Ensure we have exactly 11
    hex_positions = hex_positions[:11]
    
    inner = [[float(x), float(y), 0.0] for x, y in hex_positions]
    
    # Collect all vertices of all inner hexagons
    all_vertices = []
    for x, y, ang in inner:
        for k in range(6):
            theta = np.radians(60 * k) + np.radians(ang)
            vx = x + np.cos(theta)
            vy = y + np.sin(theta)
            all_vertices.append([vx, vy])
    
    all_vertices = np.array(all_vertices)
    
    # Find minimum enclosing regular hexagon
    # A regular hexagon with center c, side s, rotation theta contains point p if:
    # for all k=0..5: dot(p-c, n_k) <= s*sqrt(3)/2
    # where n_k are unit normals at angles theta + 60k
    
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_rotation = 0.0
    
    centroid = all_vertices.mean(axis=0)
    
    # Search over rotation
    for rot_deg in range(0, 61, 5):  # 0 to 60 in steps of 5
        if time.time() > deadline:
            break
        rot_rad = np.radians(rot_deg)
        
        normals = []
        for k in range(6):
            angle = rot_rad + np.radians(60 * k)
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)  # (6, 2)
        
        # Search center in a range around centroid
        cx_vals = np.linspace(centroid[0] - 3, centroid[0] + 3, 31)
        cy_vals = np.linspace(centroid[1] - 3, centroid[1] + 3, 31)
        
        for cx in cx_vals:
            if time.time() > deadline:
                break
            for cy in cy_vals:
                center = np.array([cx, cy])
                diffs = all_vertices - center
                projections = diffs @ normals.T  # (n_verts, 6)
                max_proj = np.max(projections)
                
                if max_proj < 0:
                    continue
                
                apothem = max_proj
                side = apothem * 2.0 / sqrt3
                
                if side < best_side:
                    best_side = side
                    best_center = (float(cx), float(cy))
                    best_rotation = float(rot_deg)
    
    if best_side == float('inf'):
        # Fallback
        best_side = 4.0
        best_center = (0.0, 0.0)
        best_rotation = 0.0
    
    return inner, best_center, float(best_side), float(best_rotation)
# EVOLVE-BLOCK-END

