"""OneMax direct binary construction (100 bits)."""
N = 100

# Pre-allocate the optimal result at module load time
_OPTIMAL = "1" * N

# EVOLVE-BLOCK-START
def construct():
    """Return a 100-bit binary string maximizing the number of ones."""
    return _OPTIMAL
# EVOLVE-BLOCK-END
