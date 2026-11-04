N = 600

# EVOLVE-BLOCK-START
def construct():
    # Use a checkerboard pattern to minimize the peak of the autoconvolution
    # assuming L_infinity normalization or similar metric where spreading mass
    # increases the peak interaction.
    h = [0.0] * N
    for i in range(0, N, 2):
        h[i] = 1.0
    return h
# EVOLVE-BLOCK-END
