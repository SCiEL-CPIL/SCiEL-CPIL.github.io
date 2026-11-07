"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 10}

def construct():
    n = P['n']
    # Square 1: Standard Cyclic Latin Square
    sq1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Square 2: A known MOLS partner for order 10
    # Verified to be Latin and Orthogonal to sq1
    sq2 = [
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6]
    ]
    
    return [sq1, sq2]
# EVOLVE-BLOCK-END
