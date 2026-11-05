"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate all points in the hexagon of radius 3
    # Condition: |a| <= 3, |b| <= 3, |a+b| <= 3
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                points.append([a, b])
    
    # Points to remove to achieve 76 edges
    # These form a path on the boundary: (3,0)-(2,1)-(1,2)-(0,3)-(-1,3)
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    # Filter points
    result = []
    for p in points:
        if (p[0], p[1]) not in remove_set:
            result.append(p)
            
    return result
# EVOLVE-BLOCK-END
