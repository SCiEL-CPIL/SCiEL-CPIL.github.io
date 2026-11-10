"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # A 10x3 grid (30 points) maximizes the multiplicity of specific small areas
    # compared to more square grids like 6x5.
    # Points are (x, y) where x in [0, 9] and y in [0, 2].
    return [(x, y) for x in range(10) for y in range(3)]
# EVOLVE-BLOCK-END
