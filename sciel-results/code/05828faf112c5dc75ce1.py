"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    # U: (n*m, rank) -> (4, 7)
    # Indices: i*m + j
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 0, 1, 0],   # A01
        [0, 1, 0, 0, 1, 0, 1],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    # V: (m*p, rank) -> (4, 7)
    # Indices: j*p + k
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 1],   # B01
        [0, 0, 0, 1, 0, 1, 0],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    # W: (p*n, rank) -> (4, 7)
    # Indices: k*n + i
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C10 (k=0, i=1)
        [0, 0, 1, 0, 1, 0, 0],   # C01 (k=1, i=0)
        [1, -1, 1, 0, 0, 1, 0]   # C11 (k=1, i=1)
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
