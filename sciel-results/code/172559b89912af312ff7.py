import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin from 120s limit

    # Scaled coordinates where minimum distance in index space is 1
    # 7-point flower
    pts = [
        (0.0, 0.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.5, math.sqrt(3)/2.0),
        (-0.5, math.sqrt(3)/2.0),
        (0.5, -math.sqrt(3)/2.0),
        (-0.5, -math.sqrt(3)/2.0)
    ]
    # 4 additional points to make 11
    # Extending the arms
    pts.append((2.0, 0.0))
    pts.append((-2.0, 0.0))
    pts.append((1.0, math.sqrt(3)))
    pts.append((-1.0, math.sqrt(3)))
    
    # Scale to real coordinates
    # Distance between centers must be >= sqrt(3) for side 1 hexagons
    # Add small epsilon to avoid floating point overlap errors
    scale = math.sqrt(3) + 1e-5
    
    inner = []
    for x, y in pts:
        inner.append([x * scale, y * scale, 0.0])
        
    # Outer hexagon side
    # The configuration extends to distance 2*scale from center in the x-direction (for point (2,0))
    # The inner hexagon at (2*scale, 0) has a vertex at x = 2*scale + 1
    # The outer hexagon must contain this.
    # With alignment 0, the outer hexagon vertex is at L.
    # So L must be >= 2*scale + 1.
    # Check other directions:
    # Point (1, sqrt(3)) is at index dist 2 from origin.
    # Physical dist 2*scale.
    # Inner hexagon vertex in that direction is at 2*scale + 1.
    # So L = 2*scale + 1 is the tight bound.
    
    L = 2.0 * scale + 1.0
    
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
