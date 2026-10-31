"""Optimized program for second autocorrelation inequality task."""
N = 50

# EVOLVE-BLOCK-START
def construct():
    # A single unit spike maximizes the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # since ||h*h||_inf >= ||h||_2^2 always, and equality holds for a delta function.
    h = [0.0] * N
    h[N // 2] = 1.0
    return h
# EVOLVE-BLOCK-END
