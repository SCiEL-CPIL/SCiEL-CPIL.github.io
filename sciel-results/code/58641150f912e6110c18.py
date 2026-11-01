# EVOLVE-BLOCK-START
def construct():
    """
    Returns the optimal rank-7 Strassen decomposition for 2x2 matrix multiplication.
    The decomposition is hardcoded for maximum speed and reliability.
    """
    # Rank 7 is the proven minimum for 2x2 matrix multiplication.
    # U, V, W correspond to the Strassen algorithm factors.
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],  # a
        [0, 0, 0, 0, 1, 0, 1],   # b
        [0, 1, 0, 0, 0, 1, 0],   # c
        [1, 1, 0, 1, 0, 0, -1]   # d
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # e
        [0, 0, 1, 0, 0, 1, 0],   # f
        [0, 0, 0, 1, 0, 0, 1],   # g
        [1, 0, -1, 0, 1, 0, 1]   # h
    ]
    
    # W follows the trace convention W[k*n+i, r]
    # Indices: 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C[0,0] = M1 + M4 - M5 + M7
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0] = M2 + M4
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1] = M3 + M5
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1] = M1 - M2 + M3 + M6
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
