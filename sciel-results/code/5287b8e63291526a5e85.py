N = 600

# EVOLVE-BLOCK-START
def construct():
    # Create a zero array
    f = [0.0] * N
    # Set the center interval to 1.0
    # This minimizes the "width" of the function, 
    # which minimizes the max autoconvolution relative to L-infinity norm squared.
    f[N // 2] = 1.0
    return f
# EVOLVE-BLOCK-END
