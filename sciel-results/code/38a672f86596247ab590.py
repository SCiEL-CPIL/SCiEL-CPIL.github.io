import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Calculate the side length of the outer hexagon
    # The 12 inner hexagons are arranged in a ring (Ring 2 of a hex grid)
    # Distance from center to inner hex centers: 2 * sqrt(3)
    # Max distance to inner hex vertex: 2 * sqrt(3) + 1
    # This fits exactly into an outer hexagon of side length 2 * sqrt(3) + 1
    
    R = 2.0 * math.sqrt(3.0)
    S = R + 1.0
    
    inner = []
    for k in range(N):
        angle_deg = k * 30.0
        angle_rad = math.radians(angle_deg)
        x = R * math.cos(angle_rad)
        y = R * math.sin(angle_rad)
        # Inner hexagons are aligned with the outer hexagon (angle 0)
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    side = S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
