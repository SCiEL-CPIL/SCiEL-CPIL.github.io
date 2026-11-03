import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 12 unit hexagons in a compact cluster
    # Center
    points = [(0.0, 0.0)]
    
    # Ring 1 (6 hexagons)
    # Distance sqrt(3)
    d1 = math.sqrt(3)
    for i in range(6):
        angle = math.pi / 3 * i
        x = d1 * math.cos(angle)
        y = d1 * math.sin(angle)
        points.append((x, y))
        
    # Ring 2 (5 hexagons)
    # We need 5 more. The "midpoints" of the second ring are at distance 3 from center.
    # Coordinates for these 6 positions:
    # (3*sqrt(3)/2, 3/2), (0, 3), (-3*sqrt(3)/2, 3/2), 
    # (-3*sqrt(3)/2, -3/2), (0, -3), (3*sqrt(3)/2, -3/2)
    # We pick the first 5.
    ring2_points = [
        (3 * math.sqrt(3) / 2, 1.5),
        (0.0, 3.0),
        (-3 * math.sqrt(3) / 2, 1.5),
        (-3 * math.sqrt(3) / 2, -1.5),
        (0.0, -3.0)
    ]
    points.extend(ring2_points)
    
    # Total 1 + 6 + 5 = 12 points
    
    inner = []
    for x, y in points:
        # Rotate by 30 degrees to minimize the bounding box size
        # when aligned with the outer hexagon's axes
        inner.append([x, y, 30.0])
        
    # Outer hexagon
    # Side length calculated as 4.0
    # Center (0,0)
    # Rotation 0.0
    return inner, (0.0, 0.0), 4.0, 0.0
# EVOLVE-BLOCK-END
