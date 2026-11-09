"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    # OneMax: the optimal solution is all ones.
    # Since there is no constraint or hidden penalty, the maximum score (N ones)
    # is achieved by returning a string of N '1' characters.
    # This completes in O(N) time and is well within the 120s deadline.
    return "1" * N
# EVOLVE-BLOCK-END
