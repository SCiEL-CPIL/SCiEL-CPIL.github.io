"""OneMax direct binary construction (100 bits).

Constructs a 100-bit binary string maximizing ones.
The global optimum is all ones, giving a score of 100.
"""
N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a 100-bit binary string maximizing ones.
    Returns a string of N '1' characters, which is the unique global optimum
    for the OneMax objective with N bits.
    """
    return '1' * N
# EVOLVE-BLOCK-END
