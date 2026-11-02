"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the Strassen rank-7 decomposition for 2x2 matrix multiplication.
    U: (4, 7)
    V: (4, 7)
    W: (4, 7)
    """
    # Strassen coefficients
    # U rows correspond to A indices (i*m+j): 00, 01, 10, 11
    # V rows correspond to B indices (j*p+k): 00, 01, 10, 11
    # W rows correspond to C indices (k*n+i): 00, 10, 01, 11
    
    U = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (w=0: k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (w=1: k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (w=2: k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (w=3: k=1, i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
