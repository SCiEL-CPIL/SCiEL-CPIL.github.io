"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    # The theoretical minimum largest element for n=12 distinct subset sums is 2^11 = 2048.
    # The set of powers of 2 achieves this bound.
    return [1 << i for i in range(12)]
# EVOLVE-BLOCK-END
