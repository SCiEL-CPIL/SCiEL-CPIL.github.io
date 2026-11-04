import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 hexagons in a compact cluster
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # First ring (6 hexagons)
    # Distance sqrt(3)
    d = math.sqrt(3)
    for i in range(6):
        angle = i * 60 * math.pi / 180
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        inner.append([x, y, 0.0])
        
    # Second ring additions (4 hexagons) to make 11
    # Placed at axial coords (2,-1), (2,1), (-2,-1), (-2,1)
    # Cartesian conversion for flat-top hex grid:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        return x, y

    additions = [
        (2, -1),
        (2, 1),
        (-2, -1),
        (-2, 1)
    ]
    
    for q, r in additions:
        x, y = axial_to_cartesian(q, r)
        inner.append([x, y, 0.0])

    # Ensure we have exactly 11
    # The list has 1 + 6 + 4 = 11.
    
    # Calculate a safe bounding side length
    # Max X extent approx 5.33, Max Y extent approx 2.37
    # Using S=7.0 is safe and much better than 40.0
    side = 7.0
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
