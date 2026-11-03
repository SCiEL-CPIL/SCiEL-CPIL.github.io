import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define grid points for 11 hexagons
    # 3 rows, 4 columns, remove one corner to optimize bounding box
    xs = [-1.5, -0.5, 0.5, 1.5]
    ys = [-1.0, 0.0, 1.0]
    
    points = []
    for y in ys:
        for x in xs:
            # Remove (1.5, 1.0) to reduce the extent in the n1 direction
            if x == 1.5 and y == 1.0:
                continue
            points.append((x, y))
            
    # Ensure we have exactly 11 points
    # The loop generates 12 points, we skip 1, so 11 remain.
    
    # Calculate the required side length S of the outer hexagon
    # The condition for a unit hexagon centered at (x,y) to be inside
    # an outer hexagon of side S (aligned) is:
    # n_i . (x,y) <= (sqrt(3)/2) * (S - 1)
    # So S = 1 + (2/sqrt(3)) * max(n_i . p)
    
    # Normals for flat-top hexagon
    normals = [
        (1.0, 0.0),
        (0.5, np.sqrt(3)/2),
        (-0.5, np.sqrt(3)/2),
        (-1.0, 0.0),
        (-0.5, -np.sqrt(3)/2),
        (0.5, -np.sqrt(3)/2)
    ]
    
    max_dot = 0.0
    for px, py in points:
        for nx, ny in normals:
            dot = nx * px + ny * py
            if dot > max_dot:
                max_dot = dot
                
    S = 1.0 + (2.0 / np.sqrt(3)) * max_dot
    
    # Construct inner hexagons
    # Format: (x, y, angle_degrees)
    inner = []
    for px, py in points:
        inner.append([px, py, 0.0])
        
    # Outer hexagon center, side, rotation
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
