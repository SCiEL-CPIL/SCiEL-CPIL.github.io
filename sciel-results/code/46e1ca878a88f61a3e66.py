"""OneMax direct binary construction (100 bits)."""

N = 100

# EVOLVE-BLOCK-START
def construct():
    # Direct construction: the optimal 100-bit string for OneMax is all ones.
    # Using integer-to-binary conversion as a structurally distinct construction path.
    # (1 << N) - 1 yields an integer with N bits all set to 1.
    val = (1 << N) - 1
    return format(val, '0{}b'.format(N))
# EVOLVE-BLOCK-END
