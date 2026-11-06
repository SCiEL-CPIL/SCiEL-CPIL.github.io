"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    result = set()
    for i in range(1, 256):
        if i % 3 == 0:
            result.add(i)
    return sorted(result)
# EVOLVE-BLOCK-END
