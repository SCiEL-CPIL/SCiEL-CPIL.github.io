import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 inner hexagon centers in axial coordinates
    # (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, 0), (0, 2), (-2, 0), (0, -2)
    ]
    
    inner = []
    for q, r in axial_coords:
        # Convert axial to Cartesian
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        inner.append([x, y, 0.0])
        
    # Calculate the required outer side length S
    # Max distance from center to inner hexagon center is D_tip = 2 * sqrt(3)
    # The constraint is dominated by the side normals.
    # S * sqrt(3) / 2 >= D_tip * cos(30) + 1
    # S * sqrt(3) / 2 >= D_tip * sqrt(3) / 2 + 1
    # S >= D_tip + 2/sqrt(3)
    
    D_tip = 2 * math.sqrt(3)
    S = D_tip + 2 / math.sqrt(3)
    
    # Add a small safety margin
    S += 1e-6
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
