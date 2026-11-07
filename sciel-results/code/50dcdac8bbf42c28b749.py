"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Return a single valid Latin square of order 10
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [square]
# EVOLVE-BLOCK-END
