"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen decomposition of rank 7
    n, m, p = DIMENSIONS
    rank = 7
    
    # U shape (n*m, rank) = (4, 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U = [
        [1, 0, 1, 0, 1, -1, 0], # a
        [0, 0, 0, 0, 1, 0, 1],  # b
        [0, 1, 0, 0, 0, 1, 0],  # c
        [1, 1, 0, 1, 0, 0, -1]  # d
    ]
    
    # V shape (m*p, rank) = (4, 7)
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V = [
        [1, 1, 0, 0, 0, 1, 0],  # e
        [0, 0, 1, -1, 0, 1, 0], # f
        [0, 0, 0, 1, 0, 0, 1],  # g
        [1, 0, -1, 0, 1, 0, 1]  # h
    ]
    
    # W shape (p*n, rank) = (4, 7)
    # Indices: 0:(k=0,i=0)->C00, 1:(k=0,i=1)->C10, 2:(k=1,i=0)->C01, 3:(k=1,i=1)->C11
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
