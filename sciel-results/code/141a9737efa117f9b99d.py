"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Use a deterministic hash to randomize the greedy order
    # This avoids lexicographic bias which often leads to suboptimal cap sets
    b = bytes(vector)
    return zlib.crc32(b)
# EVOLVE-BLOCK-END
