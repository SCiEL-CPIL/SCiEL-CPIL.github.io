import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Calculate optimal side length
    # L_scaled = 2
    # S = L_scaled / sqrt(3) + 2/3
    S = 2.0 / math.sqrt(3) + 2.0 / 3.0
    
    # Define 12 centers in axial coordinates (q, r)
    # 7 central + 5 from the next ring
    axial_points = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (1, 1),
        (-1, -1),
        (1, -2),
        (-1, 2),
        (2, -1)
    ]
    
    inner = []
    for q, r in axial_points:
        x = q + r / 2.0
        y = r * math.sqrt(3) / 2.0
        # All hexagons aligned with outer (angle 0)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
