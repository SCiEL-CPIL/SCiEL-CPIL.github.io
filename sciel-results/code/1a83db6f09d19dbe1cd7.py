"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a hardcoded set of 32 integer pairs [a, b] on the triangular lattice
    that achieves the maximum possible number of unit-distance pairs (76).
    
    The configuration is derived from the hexagon of radius 3 (37 points) by
    removing a specific path of 5 points on the boundary to minimize the loss
    of edges, resulting in exactly 76 edges.
    """
    # Generate all points in the hexagon of radius 3
    # Condition: |a| <= 3, |b| <= 3, |a+b| <= 3
    points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if abs(a) <= 3 and abs(b) <= 3 and abs(a + b) <= 3:
                points.append([a, b])
    
    # Points to remove to achieve 76 edges.
    # These 5 points form a contiguous path on the boundary of the hexagon.
    # Removing them breaks the cycle but preserves the high internal connectivity.
    remove_set = {
        (3, 0), 
        (2, 1), 
        (1, 2), 
        (0, 3), 
        (-1, 3)
    }
    
    # Filter points to get the final 32-point set
    result = []
    for p in points:
        if (p[0], p[1]) not in remove_set:
            result.append(p)
            
    return result
# EVOLVE-BLOCK-END
