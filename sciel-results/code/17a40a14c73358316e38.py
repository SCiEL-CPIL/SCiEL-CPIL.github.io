N = 100

# EVOLVE-BLOCK-START
def construct():
    # Create a 100-bit integer with all bits set to 1
    # (1 << N) - 1 gives an integer with N bits all set to 1
    val = (1 << N) - 1
    # Format the integer as a binary string of length N, padded with zeros on the left
    # Since all bits are 1, padding doesn't change the content, but ensures length N
    return format(val, '0{}b'.format(N))
# EVOLVE-BLOCK-END
