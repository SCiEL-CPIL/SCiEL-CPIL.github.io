import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define axial coordinates for 11 unit hexagons
    # This configuration is a 7-hexagon flower plus 4 additional hexagons
    # placed to maintain a compact, roughly hexagonal shape.
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, 0), (-2, 1), (-2, 2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to Cartesian for pointy-top hexagons
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 degrees
        inner.append([x, y, 0.0])
        
    # The outer hexagon must contain these.
    # Based on analysis, the required side length is approx 4.464.
    # We use 4.5 to be safe.
    outer_side = 4.5
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
