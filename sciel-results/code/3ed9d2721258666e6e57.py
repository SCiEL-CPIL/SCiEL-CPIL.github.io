N = 100

# EVOLVE-BLOCK-START
def construct():
    # Generate a 100-bit integer with all bits set to 1
    # (1 << 100) creates a 1 followed by 100 zeros, subtracting 1 yields 100 ones
    bitmask = (1 << N) - 1
    # Convert to binary string representation
    # bin(bitmask) returns '0b111...1'
    # [2:] strips the '0b' prefix
    return bin(bitmask)[2:]
# EVOLVE-BLOCK-END
