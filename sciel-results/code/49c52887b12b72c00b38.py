"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Hardcoded L2 orthogonal to L1
    L2 = [
        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
        [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
        [2, 4, 6, 8, 0, 1, 3, 5, 7, 9],
        [3, 5, 7, 9, 1, 2, 4, 6, 8, 0],
        [4, 6, 8, 0, 2, 3, 5, 7, 9, 1],
        [5, 7, 9, 1, 3, 4, 6, 8, 0, 2],
        [6, 8, 0, 2, 4, 5, 7, 9, 1, 3],
        [7, 9, 1, 3, 5, 6, 8, 0, 2, 4],
        [8, 0, 2, 4, 6, 7, 9, 1, 3, 5],
        [9, 1, 3, 5, 7, 8, 0, 2, 4, 6]
    ]
    
    # Ensure types are correct (int, list)
    # L1 is already list of lists of ints.
    # L2 is list of lists of ints.
    
    return [L1, L2]
# EVOLVE-BLOCK-END
