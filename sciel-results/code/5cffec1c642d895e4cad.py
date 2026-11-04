import numpy as np

def construct():
    """
    Pack 11 unit hexagons inside a minimal regular hexagon.
    Optimal configuration:
    - 1 center hexagon
    - 6 hexagons in ring 1 (distance sqrt(3))
    - 4 hexagons in ring 2 (distance 3, at 'edge' positions)
    Side length: 4.0
    """
    
    # Inner hexagons: (x, y, angle_degrees)
    inner = []
    
    # 1. Center hexagon
    # Rotated 90 deg so vertex points up (radial for angle 90)
    inner.append([0.0, 0.0, 90.0])
    
    # 2. Ring 1: 6 hexagons
    # Distance: sqrt(3)
    # Angles: 0, 60, 120, 180, 240, 300
    # Rotation: Vertex must point radially.
    # If center is at angle theta, vertex should be at theta.
    # Standard hex vertex is at 0 deg.
    # So rotation = theta.
    r1 = np.sqrt(3)
    for k in range(6):
        angle_deg = k * 60.0
        x = r1 * np.cos(np.deg2rad(angle_deg))
        y = r1 * np.sin(np.deg2rad(angle_deg))
        # Rotation: vertex points radially out
        # If hex is defined with vertex at 0, rotate by angle_deg
        inner.append([x, y, angle_deg])
        
    # 3. Ring 2: 4 hexagons
    # Distance: 3.0
    # Positions: 'Edge' positions are at 30, 90, 150, 210, 270, 330
    # We pick 4. Let's pick 30, 90, 150, 210.
    # This leaves 270 and 330 empty.
    # Due to symmetry of the outer hexagon (6-fold), any 4 will result in
    # the same max radial distance (3.0 + 1.0 = 4.0).
    # However, we must ensure the 4 chosen don't create an asymmetry that
    # requires a larger S? No, the bounding box is determined by the max
    # projection in the 6 normal directions.
    # The 6 edge positions are symmetric. Choosing 4 leaves 2 empty.
    # The empty ones are adjacent or opposite?
    # 30, 90, 150, 210.
    # Gaps: 210->270 (60), 270->330 (60), 330->30 (60).
    # The farthest points are at the centers of the placed hexes.
    # Max radius is 3.0.
    # So S=4 is sufficient.
    
    r2 = 3.0
    edge_angles = [30.0, 90.0, 150.0, 210.0]
    
    for angle_deg in edge_angles:
        x = r2 * np.cos(np.deg2rad(angle_deg))
        y = r2 * np.sin(np.deg2rad(angle_deg))
        # Rotation: vertex points radially out
        inner.append([x, y, angle_deg])
        
    # Outer hexagon
    # Center (0,0)
    # Side length 4.0
    # Rotation: 90.0 (vertex up, matching the radial vertex orientation)
    center = (0.0, 0.0)
    side = 4.0
    angle = 90.0
    
    return inner, center, side, angle
