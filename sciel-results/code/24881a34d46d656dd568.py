import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Configuration: 2 rows of 6 hexagons
    # Orientation: Flat Top (angle 0)
    # Hexagon side length: 1
    
    # Flat Top Hexagon vertices (relative to center):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # Width: 2, Height: sqrt(3)
    
    # Grid parameters for Flat Top
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)
    # Horizontal offset for next row: 0.75
    
    sqrt3 = math.sqrt(3)
    
    inner = []
    
    # Row 0: y = 0
    for i in range(6):
        x = i * 1.5
        y = 0.0
        inner.append([x, y, 0.0])
        
    # Row 1: y = sqrt(3)
    for i in range(6):
        x = i * 1.5 + 0.75
        y = sqrt3
        inner.append([x, y, 0.0])
        
    # Calculate bounding box of the inner shape to center it
    min_x = 0.0
    max_x = 0.0
    min_y = 0.0
    max_y = 0.0
    
    # We need the actual vertices to find the exact bounding box
    # But we already calculated it analytically:
    # Min X: -1 (from first hex in row 0)
    # Max X: 9.25 (from last hex in row 1: 8.25 + 1)
    # Min Y: -sqrt(3)/2 (from row 0)
    # Max Y: 1.5 * sqrt(3) (from row 1: sqrt(3) + sqrt(3)/2)
    
    min_x = -1.0
    max_x = 9.25
    min_y = -sqrt3 / 2.0
    max_y = 1.5 * sqrt3
    
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    
    # Shift all inner hexagons to center the assembly at (0,0)
    for i in range(len(inner)):
        inner[i][0] -= center_x
        inner[i][1] -= center_y
        
    # Outer hexagon
    # Side length S
    # We calculated S = 5.625 for this configuration
    side = 5.625
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    return inner, outer_center, side, outer_angle
# EVOLVE-BLOCK-END
