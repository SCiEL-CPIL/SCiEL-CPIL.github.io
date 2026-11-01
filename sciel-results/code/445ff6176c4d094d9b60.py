"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Uses a permuted column order (reversed) of the standard Strassen factors.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    # Standard Strassen factors (columns correspond to the 7 products)
    # We reverse the column order to create a distinct but equivalent representation.
    
    U_orig = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_orig = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_orig = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Reverse columns
    U = [list(row[::-1]) for row in U_orig]
    V = [list(row[::-1]) for row in V_orig]
    W = [list(row[::-1]) for row in W_orig]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
