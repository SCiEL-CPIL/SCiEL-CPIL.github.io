"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    # Based on the analysis, N=31 is achievable with x_i = 225 + i
    # x_1 = 226, x_31 = 256
    # This satisfies the condition that subset sums of different cardinalities are disjoint.
    n = 31
    start = 225
    return [start + i for i in range(1, n + 1)]
# EVOLVE-BLOCK-END
