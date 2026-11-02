import math

N = 11

def construct():
    # Axial coordinates for a compact cluster of 11 hexagons
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    inner = []
    max_x = 0
    max_y = 0
    max_r = 0
    
    for q, r in axial_coords:
        x = 1.5 * q
        y = math.sqrt(3) / 2 * (r + q / 2)
        inner.append([x, y, 0.0])
        
        # Calculate the extent of this hexagon
        # The hexagon has vertices at distance 1 from center.
        # So the max extent in any direction is 1.
        # We need to find the minimum side length S of a regular hexagon
        # centered at (0,0) that contains all these hexagons.
        
    # To find the side length S, we can check the distance of each vertex
    # of each inner hexagon from the center.
    # The container is a regular hexagon of side S.
    # The distance from center to the side is apothem = S * sqrt(3) / 2.
    # The distance from center to a vertex is S.
    # A point (x,y) is inside a regular hexagon of side S (rotated 0 deg) if:
    # |x| <= S * 3/2 ? No.
    # Let's just check the 6 vertices of the container.
    # Container vertices: (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # Actually, simpler: for each inner hexagon, check its 6 vertices.
    # If a vertex is outside the container, S is too small.
    # We can binary search S or just calculate the max required S.
    
    # For a point (px, py) to be inside a hexagon of side S (centered at 0,0, flat top? pointy top?)
    # Let's assume the container is aligned with the inner hexagons (angle 0).
    # The container's "radius" (center to vertex) is S.
    # The container's "apothem" (center to side midpoint) is S * sqrt(3)/2.
    # The hexagon is the intersection of 6 half-planes.
    # The normals are at angles 0, 60, 120, 180, 240, 300.
    # For a point (px, py), the projection onto the normal at angle theta is px*cos(theta) + py*sin(theta).
    # This projection must be <= S * sqrt(3)/2.
    # So S >= (px*cos(theta) + py*sin(theta)) / (sqrt(3)/2) for all theta.
    
    S = 0.0
    for x, y, _ in inner:
        # Check 6 vertices of this unit hexagon
        for k in range(6):
            angle = math.radians(60 * k)
            vx = x + math.cos(angle)
            vy = y + math.sin(angle)
            for theta_deg in range(0, 360, 60):
                theta = math.radians(theta_deg)
                proj = vx * math.cos(theta) + vy * math.sin(theta)
                required_S = proj / (math.sqrt(3) / 2)
                if required_S > S:
                    S = required_S
                    
    return inner, (0.0, 0.0), S, 0.0
