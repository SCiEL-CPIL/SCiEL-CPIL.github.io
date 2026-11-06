"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    t0 = time.monotonic()
    limit = t0 + 145.0
    # Use integer arithmetic to construct the all-ones bitstring
    mask = (1 << N) - 1
    s = bin(mask)[2:]
    if time.monotonic() < limit:
        return s
    return s
# EVOLVE-BLOCK-END
