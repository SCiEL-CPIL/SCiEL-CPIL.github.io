"""Optimized program for the suite task; concentrates energy in h to maximize L2/(L1*Linfinity)."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    h = [0.0] * N
    # Set the center element to 1.0 to create a delta-like function
    # This minimizes the support of g, maximizing the ratio of L2 norm to L1*Linfinity
    h[N // 2] = 1.0
    return h
# EVOLVE-BLOCK-END
