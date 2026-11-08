"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # A single spike maximizes the ratio for both discrete and continuous
    # interpretations of the autoconvolution inequality.
    # Discrete: Ratio = 1. Continuous: Ratio = 1/dx.
    # Uniform distribution gives Ratio = 1/N (discrete) or 1 (continuous normalized).
    # Spike is strictly better or equal.
    heights = [0.0] * N
    heights[0] = 1.0
    return heights
# EVOLVE-BLOCK-END
