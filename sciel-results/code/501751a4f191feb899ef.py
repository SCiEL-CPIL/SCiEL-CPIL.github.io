"""
OneMax (100 bits) direct binary construction.
The objective is to maximize the count of '1' characters in a 100-bit string.
The global optimum is the all-ones string, which yields a score of 100.
"""
import time

N = 100
DEADLINE_SECONDS = 150.0

# EVOLVE-BLOCK-START
def construct():
    # The OneMax objective is separable and linear: f(x) = sum_i x_i with x_i in {0,1}.
    # The unique global maximizer is x_i = 1 for all i.
    # Construct the optimal string directly in O(N) time.
    return "1" * N
# EVOLVE-BLOCK-END
