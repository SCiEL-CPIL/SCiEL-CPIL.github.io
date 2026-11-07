"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    # Return a single valid Latin square: L[i][j] = (i + j) % n
    # This is a standard cyclic Latin square.
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [square]
# EVOLVE-BLOCK-END
