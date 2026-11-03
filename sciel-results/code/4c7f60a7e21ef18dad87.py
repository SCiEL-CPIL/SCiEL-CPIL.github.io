import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # 3x4 grid of hexagons
    # Rows at y=0, sqrt(3), 2*sqrt(3)
    # Cols at x=-3,-1,1,3 (rows 0,2) and x=-2,0,2,4 (row 1)
    
    sqrt3 = np.sqrt(3)
    
    centers = []
    # Row 0
    for x in [-3, -1, 1, 3]:
        centers.append([x, 0.0])
    # Row 1
    for x in [-2, 0, 2, 4]:
        centers.append([x, sqrt3])
    # Row 2
    for x in [-3, -1, 1, 3]:
        centers.append([x, 2*sqrt3])
        
    centers = np.array(centers)
    
    # Center the cluster at the origin
    com_x = np.mean(centers[:, 0])
    com_y = np.mean(centers[:, 1])
    centers[:, 0] -= com_x
    centers[:, 1] -= com_y
    
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    # Estimated minimum side length is ~5.33
    # Use 5.4 for safety margin
    side = 5.4
    
    return inner, (0.0, 0.0), side, 0.0
# EVOLVE-BLOCK-END
