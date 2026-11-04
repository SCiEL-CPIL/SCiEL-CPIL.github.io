import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # We need to pack N=11 unit hexagons into the smallest outer hexagon.
    # The number of unit hexagons that fit in a side-S hexagon is 3*S*(S-1) + 1.
    # S=1 -> 1
    # S=2 -> 7
    # S=3 -> 19
    # Since 11 > 7, we need S=3.
    
    S = 3
    radius = S - 1  # Axial coordinate radius
    
    # Generate axial coordinates (q, r) for a hexagon of given radius
    coords = []
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            coords.append((q, r))
            
    # We have enough coordinates (19 for S=3). We only need N=11.
    # Convert axial to Cartesian.
    # Using pointy-top hexagons:
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    # This creates a grid where adjacent hexagons touch.
    
    inner = []
    for i in range(N):
        q, r = coords[i]
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        # Angle 0.0 means the hexagon is oriented with a point at the top/bottom (or side, depending on convention)
        # Since the outer hexagon is also angle 0.0, they align perfectly for tiling.
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    center = (0.0, 0.0)
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
