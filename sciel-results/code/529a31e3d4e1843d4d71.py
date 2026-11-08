"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # Use a single spike to maximize L2^2 / (L1 * sup_autoconv)
    # For a single 1: L2^2=1, L1=1, autoconv_sup=1, ratio=1
    # This should outperform the uniform [1.0]*50 which has ratio ~0.02
    h = [0.0] * N
    h[N // 2] = 1.0
    return h
# EVOLVE-BLOCK-END
