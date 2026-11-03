import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Theoretical minimum side length based on area: S = sqrt(12) = 2*sqrt(3)
    # We use a 3x4 staggered grid packing which fits exactly in width and has slack in height.
    S = 2.0 * math.sqrt(3)
    
    # Outer hexagon: center (0,0), side S, angle 0
    outer_center = (0.0, 0.0)
    outer_side = S
    outer_angle = 0.0
    
    inner = []
    
    # Parameters for the 3x4 grid
    # Hexagon width (distance between opposite sides) = sqrt(3)
    # Hexagon height (distance between opposite vertices) = 2.0
    # Distance between centers in same row = sqrt(3)
    # Vertical distance between rows = 1.5
    
    dx = math.sqrt(3)
    dy = 1.5
    
    # Row 0 (bottom)
    # 4 hexagons, centered horizontally
    # x offsets: -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    y0 = -1.5
    for i in range(4):
        x = (i - 1.5) * dx
        inner.append([x, y0, 0.0])
        
    # Row 1 (middle)
    # Staggered by dx/2
    y1 = 0.0
    for i in range(4):
        x = (i - 1.5) * dx + dx/2.0
        inner.append([x, y1, 0.0])
        
    # Row 2 (top)
    y2 = 1.5
    for i in range(4):
        x = (i - 1.5) * dx
        inner.append([x, y2, 0.0])
        
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
