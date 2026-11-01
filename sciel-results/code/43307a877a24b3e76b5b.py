import numpy as np

N = 11

def construct():
    # Generate 11 positions in a hexagonal grid
    # Axial coordinates (q, r)
    # Distance from center in hex units is max(abs(q), abs(r), abs(q+r))
    
    positions_axial = []
    
    # Add center
    positions_axial.append((0, 0))
    
    # Add neighbors in rings
    # Ring 1
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    positions_axial.extend(ring1)
    
    # Need 4 more from Ring 2
    # Ring 2 coordinates:
    # (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    # We pick 4 that are spread out to maintain symmetry/compactness
    # (2, 0), (0, -2), (-2, 0), (0, 2) -> This makes a cross, might be large.
    # Let's just take the first 4 in a spiral order to be safe.
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), 
        (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    ]
    positions_axial.extend(ring2_candidates[:4])
    
    # Ensure we have exactly N
    positions_axial = positions_axial[:N]
    
    # Convert axial to Cartesian
    # Assuming pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    # Calculate vertices for all inner hexagons
    # Pointy-top hexagon vertices are at angles 30, 90, 150, 210, 270, 330 degrees
    # Radius = 1
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    vertices = []
    for cx, cy in centers:
        for a in angles:
            vx = cx + np.cos(a)
            vy = cy + np.sin(a)
            vertices.append((vx, vy))
    
    vertices = np.array(vertices)
    
    # Determine the minimum enclosing regular hexagon side length
    # The outer hexagon is also pointy-top (angle 0).
    # Its sides are perpendicular to angles 0, 60, 120, 180, 240, 300.
    # The apothem A is the distance from center to side.
    # Side length S = 2 * A / sqrt(3)
    # We need to find the maximum projection of any vertex onto the 6 normal directions.
    
    normal_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    max_apothem = 0.0
    
    for na in normal_angles:
        nx = np.cos(na)
        ny = np.sin(na)
        # Project all vertices onto this normal
        projections = vertices[:, 0] * nx + vertices[:, 1] * ny
        # The distance to the side is the max absolute value of the projection
        # Since the hexagon is centered at 0, the side is at distance A.
        # All points must have |projection| <= A
        current_max = np.max(np.abs(projections))
        if current_max > max_apothem:
            max_apothem = current_max
            
    side_length = 2.0 * max_apothem / sqrt3
    
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side_length, angle

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import numpy as np

    N = 11

    # Generate 11 positions in a hexagonal grid
    positions_axial = []
    positions_axial.append((0, 0))
    
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    positions_axial.extend(ring1)
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), 
        (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    ]
    positions_axial.extend(ring2_candidates[:4])
    
    positions_axial = positions_axial[:N]
    
    sqrt3 = np.sqrt(3)
    centers = []
    for q, r in positions_axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    vertices = []
    for cx, cy in centers:
        for a in angles:
            vx = cx + np.cos(a)
            vy = cy + np.sin(a)
            vertices.append((vx, vy))
    
    vertices = np.array(vertices)
    
    normal_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    max_apothem = 0.0
    
    for na in normal_angles:
        nx = np.cos(na)
        ny = np.sin(na)
        projections = vertices[:, 0] * nx + vertices[:, 1] * ny
        current_max = np.max(np.abs(projections))
        if current_max > max_apothem:
            max_apothem = current_max
            
    side_length = 2.0 * max_apothem / sqrt3
    
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side_length, angle
# EVOLVE-BLOCK-END
