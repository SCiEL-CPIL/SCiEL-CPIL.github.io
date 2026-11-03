import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3x4 honeycomb grid
    # Row 0 (bottom): 4 hexagons
    # Row 1 (middle): 4 hexagons
    # Row 2 (top): 4 hexagons
    
    # Spacing
    dx = np.sqrt(3)
    dy = 1.5
    
    # Generate centers
    centers = []
    
    # Row 0 (y = -1.5)
    # To center the 3x4 block, we need to shift x appropriately.
    # Row 0 x: 0, dx, 2dx, 3dx
    # Row 1 x: dx/2, 1.5dx, 2.5dx, 3.5dx
    # Row 2 x: 0, dx, 2dx, 3dx
    
    # Center of the block in x:
    # Row 0 center: 1.5dx
    # Row 1 center: 2dx
    # Row 2 center: 1.5dx
    # Avg center: (1.5+2+1.5)/3 * dx = 5/3 dx
    
    # Center of the block in y:
    # -1.5, 0, 1.5 -> Avg 0
    
    shift_x = (5/3) * dx
    shift_y = 0
    
    for i in range(3):
        y = -1.5 + i * dy
        if i == 0 or i == 2:
            # Rows 0 and 2 are aligned
            for j in range(4):
                x = j * dx
                centers.append([x - shift_x, y - shift_y, 0.0])
        else:
            # Row 1 is shifted
            for j in range(4):
                x = (j + 0.5) * dx
                centers.append([x - shift_x, y - shift_y, 0.0])
    
    inner = centers
    center = [0.0, 0.0]
    side = 4.0
    angle = 30.0 # Pointy top to maximize width for the 3x4 cluster
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
