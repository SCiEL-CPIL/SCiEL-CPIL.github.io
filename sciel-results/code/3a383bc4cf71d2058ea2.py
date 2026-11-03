import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the 12 hexagons in axial coordinates (q, r)
    # 7 inner + 5 outer as derived
    coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        
        (-2, 0),
        (-2, 1),
        (0, -2),
        (1, -2),
        (-1, -1)
    ]
    
    inner = []
    for q, r in coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Center of outer hexagon
    center = (0.0, 0.0)
    # Side length
    side = 4.0
    # Rotation
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
