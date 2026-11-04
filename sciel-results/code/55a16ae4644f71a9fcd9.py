"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    h = [0.0] * N
    h[N // 2] = 1.0
    return h
# EVOLVE-BLOCK-END
