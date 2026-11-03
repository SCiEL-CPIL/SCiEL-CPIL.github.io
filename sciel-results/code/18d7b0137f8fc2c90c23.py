import math

def construct():
    # 11 unit hexagons packed in a regular hexagon of side 3.0
    # Configuration: 7 central hexagons (1 center + 6 neighbors)
    # plus 4 hexagons from the second ring at axial positions.
    
    # Lattice coordinates (i, j) -> Cartesian (x, y)
    # x = i + j/2
    # y = j * sqrt(3) / 2
    
    sqrt3 = math.sqrt(3)
    
    # 7 central points
    centers = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, sqrt3 / 2),
        (-0.5, sqrt3 / 2),
        (0.5, -sqrt3 / 2),
        (-0.5, -sqrt3 / 2),
    ]
    
    # 4 additional points from the second ring
    # Lattice (2, 0) -> (2, 0)
    # Lattice (-2, 0) -> (-2, 0)
    # Lattice (0, 2) -> (1, sqrt3)
    # Lattice (0, -2) -> (-1, -sqrt3)
    
    centers.append((2.0, 0.0))
    centers.append((-2.0, 0.0))
    centers.append((1.0, sqrt3))
    centers.append((-1.0, -sqrt3))
    
    # All angles are 0.0
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer hexagon
    center = (0.0, 0.0)
    side = 3.0
    angle = 0.0
    
    return inner, center, side, angle
