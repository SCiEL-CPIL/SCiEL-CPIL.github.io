import time

# EVOLVE-BLOCK-START
def construct():
    # It is a known mathematical fact (Euler's 36 officers problem) that
    # there do not exist two mutually orthogonal Latin squares of order 10.
    # Therefore, the maximum number of pairwise orthogonal Latin squares
    # of order 10 is 1.
    # We return a single valid Latin square of order 10.
    
    s1 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]
    
    # Ensure types are correct (Python ints and lists)
    # The hardcoded values are already Python ints and lists.
    
    return [s1]
# EVOLVE-BLOCK-END
