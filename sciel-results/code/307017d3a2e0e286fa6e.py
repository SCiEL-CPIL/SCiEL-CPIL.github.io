"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # Use the Paley tournament on 67 vertices.
    # 67 is a prime with 67 % 4 == 3.
    # The Paley tournament is defined by i -> j iff (j - i) % p is a quadratic residue.
    # It is a known result that the Paley tournament on 67 vertices satisfies the 
    # condition that every set of 4 vertices has a common dominator.
    
    p = 67
    # Compute quadratic residues modulo p
    squares = {x * x % p for x in range(1, p)}
    
    # Construct the orientation list
    # One bit per pair (i, j), i < j, in lexicographic order.
    # 0 means i -> j, 1 means j -> i.
    
    # Vectorized construction for speed
    i_vals, j_vals = np.triu_indices(p, k=1)
    diffs = (j_vals - i_vals) % p
    
    # Create a boolean mask for quadratic residues
    # We need to check if each diff is in squares
    # Since p is small (67), we can create a lookup array
    lookup = np.zeros(p, dtype=bool)
    for s in squares:
        lookup[s] = True
    
    # 0 means i -> j (diff is square), 1 means j -> i (diff is not square)
    orientation = (~lookup[diffs]).astype(int)
    
    return {'n': p, 'orientation': orientation.tolist()}
# EVOLVE-BLOCK-END
