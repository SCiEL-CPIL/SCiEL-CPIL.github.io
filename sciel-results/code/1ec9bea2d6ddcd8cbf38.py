N = 600

# EVOLVE-BLOCK-START
def construct():
    # The seed program likely used a constant function which minimizes
    # the ratio max(f*f) / ||f||_1^2.
    # The current program uses a delta function, which maximizes this ratio.
    # We revert to a constant function to achieve a high score.
    f = [1.0] * N
    return f
# EVOLVE-BLOCK-END
