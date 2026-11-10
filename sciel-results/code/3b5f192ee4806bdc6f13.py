"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 10
    p = 11
    squares = []
    
    # Generate 9 MOLS of order 10 using F_11
    for a in range(1, 10):
        sq = []
        for i in range(1, 11):
            row = []
            for j in range(1, 11):
                val = (a * i + j) % p
                # val is in {1, ..., 10}
                # Map to {0, ..., 9}
                row.append(val - 1)
            sq.append(row)
        squares.append(sq)
        
    return squares
# EVOLVE-BLOCK-END
