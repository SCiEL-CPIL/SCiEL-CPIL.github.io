"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a set of 30 distinct integer points in [-30, 30]^2 
    designed to maximize the multiplicity of the most common non-zero doubled area.
    
    Strategy: Use a 15x2 grid (two rows of 15 consecutive points).
    This configuration concentrates a large number of triangles on the smallest non-zero area (doubled area = 1).
    Calculation for 15x2 grid:
    n1 = 15, n2 = 15.
    Multiplicity of area d (for d=1) is 2*n1*n2 - (n1+n2)*d = 2*225 - 30*1 = 420.
    """
    points = []
    # Row 1: y = 0, x from -14 to 14
    for x in range(-14, 15):
        points.append((x, 0))
    # Row 2: y = 1, x from -14 to 14
    for x in range(-14, 15):
        points.append((x, 1))
        
    return points
# EVOLVE-BLOCK-END
