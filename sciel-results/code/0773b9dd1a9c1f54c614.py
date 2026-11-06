"""
Rewritten program for OneMax (100 bits).
Uses a direct analytical solution since the objective is linear and separable.
"""
N = 100

# EVOLVE-BLOCK-START
def construct():
    # The OneMax objective is to maximize the number of 1s in a binary string of length N.
    # The global optimum is the string consisting entirely of '1's.
    # This is a direct analytical solution.
    return "1" * N
# EVOLVE-BLOCK-END
