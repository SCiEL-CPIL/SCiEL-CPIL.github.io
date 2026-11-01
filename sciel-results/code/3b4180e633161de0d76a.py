import math

def construct():
    # Axial coordinates (q, r) for the 11 unit hexagons
    # This configuration has a bounding hexagon of side S=3 (in physical units for centers)
    # which results in an outer hexagon side length L = 3 + 4/sqrt(3)
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (1, 1), (-1, -1),
        (2, -1), (-2, 1)
    ]
    
    # Convert axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = math.sqrt(3)
    inner = []
    for q, r in axial_coords:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        # Angle is 0.0 as we align with the outer hexagon
        inner.append([x, y, 0.0])
    
    # Calculate outer side length
    # S is the side length of the hexagon containing the centers
    # S = 3.0
    S = 3.0
    # L = S + 4/sqrt(3)
    L = S + 4.0 / sqrt3
    
    return inner, (0.0, 0.0), L, 0.0
