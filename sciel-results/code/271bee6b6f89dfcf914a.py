import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 centers for unit hexagons
    # They are arranged in a hexagonal lattice pattern
    # 7 inner points + 5 outer points (selected to minimize bounding box)
    
    sqrt3 = math.sqrt(3)
    
    centers = [
        [0.0, 0.0],
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [1.5*sqrt3, -1.5],
        [0.0, -3.0],
        [-1.5*sqrt3, -1.5],
        [-1.5*sqrt3, 1.5],
        [1.5*sqrt3, 1.5]
    ]
    
    # The centers fit inside a hexagon of side length 3.0
    # The outer hexagon must be side length 3.0 + 1.0 = 4.0
    # to contain the unit hexagons (which have inradius sqrt(3)/2)
    # assuming alignment.
    
    inner = [[x, y, 0.0] for x, y in centers]
    center = (0.0, 0.0)
    side = 4.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
