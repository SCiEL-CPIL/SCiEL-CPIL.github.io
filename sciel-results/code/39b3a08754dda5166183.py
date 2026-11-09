"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Construct a single Latin square using the cyclic group Z_n
    # A[i][j] = (i + j) % n
    sq = [[(i + j) % n for j in range(n)] for i in range(n)]
    return [sq]
# EVOLVE-BLOCK-END
