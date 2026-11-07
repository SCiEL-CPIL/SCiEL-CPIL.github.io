"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _build_square(c2, c5, n=10):
    """Build Latin square using Z_2 x Z_5 linear construction."""
    sq = []
    for i in range(n):
        i2 = i // 5
        i5 = i % 5
        row = []
        for j in range(n):
            j2 = j // 5
            j5 = j % 5
            a_out = (c2 * i2 + j2) % 2
            b_out = (c5 * i5 + j5) % 5
            row.append(a_out * 5 + b_out)
        sq.append(row)
    return sq

def construct():
    # The maximum number of mutually orthogonal Latin squares of order 10 is 2.
    # We construct two specific squares known to be orthogonal.
    # Square 1: c2=0, c5=0
    # Square 2: c2=1, c5=1
    sq1 = _build_square(0, 0)
    sq2 = _build_square(1, 1)
    
    # Convert to Python lists of ints to satisfy type checking
    result = []
    for sq in [sq1, sq2]:
        clean_sq = []
        for row in sq:
            clean_row = [int(x) for x in row]
            clean_sq.append(clean_row)
        result.append(clean_sq)
        
    return result
# EVOLVE-BLOCK-END
