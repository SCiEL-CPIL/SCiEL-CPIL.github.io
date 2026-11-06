"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set of integers from 129 to 256 is valid.
    # For any a, b, c in this set with a < b <= c:
    # a >= 129, b >= 130, c >= 130.
    # b + c >= 260.
    # The smallest multiple of a greater than b+c is 2a (since b+c > a).
    # If 2a = b+c, then c = 2a - b.
    # Since b > a, c < a. But we require c >= b > a. Contradiction.
    # Thus, no such triple exists.
    # Size is 256 - 129 + 1 = 128.
    return list(range(129, 257))
# EVOLVE-BLOCK-END
