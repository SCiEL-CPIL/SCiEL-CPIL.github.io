"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    # Indices:
    # U rows: i*m+j (0:a11, 1:a12, 2:a21, 3:a22)
    # V rows: j*p+k (0:b11, 1:b12, 2:b21, 3:b22)
    # W rows: k*n+i (0:c11, 1:c21, 2:c12, 3:c22)
    
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W = [
        [1, 0, -1, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 0, 0, 0, 1, 0]
    ]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
