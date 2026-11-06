"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use a 2x15 grid to maximize the multiplicity of area 1.
    # Points on y=0 and y=1 with x from 0 to 14.
    points = []
    for x in range(15):
        points.append((x, 0))
        points.append((x, 1))
    return points
# EVOLVE-BLOCK-END
