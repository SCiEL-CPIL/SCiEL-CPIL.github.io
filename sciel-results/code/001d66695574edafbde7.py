import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 100-bit binary string maximizing the number of ones.
    
    For the OneMax problem with a linear objective function where each '1' contributes
    positively to the score, the global optimum is a string consisting entirely of '1's.
    Therefore, we directly construct this string.
    """
    # Use bytearray for low-level construction of the optimal string.
    # 0x31 is the ASCII code for '1'.
    b = bytearray([0x31] * N)
    return b.decode('utf-8')
# EVOLVE-BLOCK-END
