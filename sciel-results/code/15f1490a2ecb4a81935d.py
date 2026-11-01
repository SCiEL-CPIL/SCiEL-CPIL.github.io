import math

def construct():
    # Define the 11 inner hexagons
    # Using axial coordinates (q, r) for the hexagonal grid
    # Center: (0,0)
    # 6 Corners of Ring 2:
    # (2,0), (1,1), (-1,2), (-2,1), (-1,-1), (1,-2)
    # 4 Sides of Ring 2 (closer to center):
    # (2,-1), (0,2), (-2,2), (-2,0)
    
    axial_coords = [
        (0, 0),
        (2, 0),
        (1, 1),
        (-1, 2),
        (-2, 1),
        (-1, -1),
        (1, -2),
        (2, -1),
        (0, 2),
        (-2, 2),
        (-2, 0)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to cartesian
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Calculate the minimal side length of the outer hexagon
    # The maximum distance from the center to any vertex of the inner hexagons
    # is determined by the "Corner" hexagons in Ring 2.
    # Distance from center to corner hex center: 2 * sqrt(3)
    # Radius of a unit hexagon: 1
    # Total radius required: 2 * sqrt(3) + 1
    side = 2 * math.sqrt(3) + 1
    
    return inner, (0.0, 0.0), side, 0.0
