"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import zlib

# EVOLVE-BLOCK-START
def priority(vector, n):
    # Combine a deterministic hash with a structural bias.
    # The bias uses a weighted sum of the vector components, which
    # may help the greedy algorithm select vectors that are less
    # likely to form 3-term arithmetic progressions with each other
    # by spreading the selection based on positional values.
    h = zlib.crc32(bytes(vector))
    bias = sum(v * (i + 1) for i, v in enumerate(vector))
    return float(h + bias * 1000)
# EVOLVE-BLOCK-END
