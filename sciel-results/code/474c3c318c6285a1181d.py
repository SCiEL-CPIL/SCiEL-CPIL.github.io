import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate optimal side length
    S = 3 + 2 * math.sqrt(3) / 3
    
    # Inner hexagons
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # Ring 1
    r1 = math.sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = r1 * math.cos(rad)
        y = r1 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # Ring 2 (4 corners)
    r2 = 3.0
    # Angles 30, 150, 210, 330
    for angle in [30.0, 150.0, 210.0, 330.0]:
        rad = math.radians(angle)
        x = r2 * math.cos(rad)
        y = r2 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # Rotate outer by 30 degrees so its normals align with 0, 60, ...
    # Inner hexagons are at 0 degrees (vertices at 0, 60, ...)
    # This configuration minimizes S as derived.
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle = 30.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
