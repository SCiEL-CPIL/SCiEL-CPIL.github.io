"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 145s limit, safe margin from 150s
    
    # The known optimal construction for this problem is the Paley tournament on p=67.
    # p must be a prime congruent to 3 mod 4.
    p = 67
    
    # Check if p is prime (67 is prime)
    # Check p % 4 == 3 (67 % 4 == 3)
    
    # Compute quadratic residues mod p
    # i -> j iff (j - i) % p is a nonzero quadratic residue
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
        
    # Construct orientation
    # n = p
    # Pairs (i, j) with i < j in lexicographic order
    # orientation bit 0 means i -> j, 1 means j -> i
    # i -> j iff (j - i) % p in squares
    
    n = p
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)
            else:
                orientation.append(1)
                
    return {'n': n, 'orientation': orientation}
# EVOLVE-BLOCK-END
