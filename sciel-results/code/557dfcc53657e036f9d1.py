# EVOLVE-BLOCK-START
def construct():
    # Pre-computed list of 32 integer pairs forming an optimal Harborth configuration
    # These are the points in the hexagon of radius 3 excluding the boundary path:
    # (3,0), (2,1), (1,2), (0,3), (-1,3)
    
    # Full hexagon R=3 points:
    # a in [-3, 3], b in [-3, 3], |a+b| <= 3
    
    points = [
        [-3, 0], [-3, 1], [-3, 2], [-3, 3],
        [-2, -1], [-2, 0], [-2, 1], [-2, 2], [-2, 3],
        [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], [-1, 3],
        [0, -3], [0, -2], [0, -1], [0, 0], [0, 1], [0, 2], [0, 3],
        [1, -3], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        [2, -3], [2, -2], [2, -1], [2, 0], [2, 1],
        [3, -3], [3, -2], [3, -1], [3, 0]
    ]
    
    # Remove the 5 points: (3,0), (2,1), (1,2), (0,3), (-1,3)
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    result = []
    for p in points:
        if (p[0], p[1]) not in remove_set:
            result.append(p)
            
    return result
# EVOLVE-BLOCK-END
