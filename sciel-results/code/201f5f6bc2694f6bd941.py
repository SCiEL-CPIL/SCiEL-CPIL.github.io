"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    n, m, p = DIMENSIONS
    rank = 7
    # U shape (n*m, rank) = (4, 7)
    # Indices: 0: A00, 1: A01, 2: A10, 3: A11
    U = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 0, 0, 0, 1, 0, 1], # A01
        [0, 1, 0, 0, 0, 1, -1],# A10
        [1, 1, 0, 1, 0, -1, 0] # A11
    ]
    
    # V shape (m*p, rank) = (4, 7)
    # Indices: 0: B00, 1: B01, 2: B10, 3: B11
    V = [
        [1, 1, 0, -1, 0, 0, 1], # B00
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [0, 0, 0, 1, 0, 0, 0],  # B10
        [1, 0, -1, 0, 1, 1, 1]  # B11
    ]
    
    # W shape (p*n, rank) = (4, 7)
    # Indices: 0: C00, 1: C10, 2: C01, 3: C11
    # Based on W[k*n+i] with n=2:
    # k=0, i=0 -> 0 (C00)
    # k=0, i=1 -> 1 (C10)
    # k=1, i=0 -> 2 (C01)
    # k=1, i=1 -> 3 (C11)
    W = [
        [1, 0, 0, -1, 1, 0, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
