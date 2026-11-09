"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

P = {'n': 10}

def construct():
    n = P['n']
    
    # Generate L1
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Generate L2: (2i + j) % n
    # Check orthogonality:
    # a = i + j
    # b = 2i + j
    # i = b - a
    # j = 2a - b
    # Since gcd(1, n) = 1, this is a bijection for n=10.
    L2 = [[(2 * i + j) % n for j in range(n)] for i in range(n)]
    
    squares = [L1, L2]
    
    # Ensure result is list of lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
        
    return result
# EVOLVE-BLOCK-END
