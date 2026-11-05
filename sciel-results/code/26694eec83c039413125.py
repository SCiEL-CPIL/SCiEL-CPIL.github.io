"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Uses Python's arbitrary-precision integer arithmetic:
    - Create a mask of 2^100 - 1 (all 100 bits set to 1)
    - Convert to binary string using bin()
    - Strip the '0b' prefix
    - Zero-pad to exactly N characters for safety
    """
    mask = (1 << N) - 1
    s = bin(mask)[2:].zfill(N)
    return s
# EVOLVE-BLOCK-END
