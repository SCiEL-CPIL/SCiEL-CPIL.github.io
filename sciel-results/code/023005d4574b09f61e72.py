"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use the Paley graph of order 29.
    # The Paley graph P_29 is a strongly regular graph with parameters (29, 14, 6, 7).
    # It is known that P_29 and its complement are K_{2,2,2}-free.
    # This certifies R(K_{2,2,2}, K_{2,2,2}) >= 29.
    
    p = 29
    # Compute non-zero quadratic residues modulo p
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    n = p
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % p
            # Color 0 if difference is a square, 1 otherwise
            # This defines the Paley graph.
            if diff in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
