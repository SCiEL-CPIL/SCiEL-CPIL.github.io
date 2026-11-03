import math

N = 12

def construct():
    # Axial coordinates for 12 hexagons in a hexagonal lattice
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 5 hexagons (selected to be compact)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to cartesian (pointy-top, side length 1)
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 assumes pointy-top orientation aligned with container
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Based on analysis, side length 5.0 is sufficient for this configuration
    center = [0.0, 0.0]
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
