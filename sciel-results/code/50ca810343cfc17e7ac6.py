"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # For 3x3 matrix multiplication, Strassen's algorithm gives rank 7
    # For 3x3x7, we split into 3x3x3 + 3x3x3 + 3x3x1
    # Using Strassen for the two 3x3x3 blocks (rank 7 each) and schoolbook for 3x3x1 (rank 9)
    # Total rank = 7 + 7 + 9 = 23
    
    # Strassen's algorithm for 3x3 matrix multiplication
    # We need to decompose the 3x3 matrix multiplication tensor into 7 rank-1 tensors
    # The standard Strassen decomposition for 3x3 uses 7 multiplications
    
    # Let's use the known rank-7 decomposition for 3x3 matrix multiplication
    # Reference: The tensor for 3x3 matrix multiplication has rank 7 (Strassen)
    
    # Strassen's 7-multiplication algorithm for 3x3 matrices:
    # We'll use the standard formulation
    
    rank = 23
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    # Strassen decomposition for first 3x3 block (columns 0-2 of B)
    # The 7 rank-1 tensors for 3x3 matrix multiplication
    # Using the standard Strassen algorithm adapted for 3x3
    
    # For 3x3, we can use the following decomposition:
    # M1 = (a11+a22+a33)(b11+b22+b33) - a11*b11 - a22*b22 - a33*b33
    # But this doesn't directly give rank 7 for the full tensor
    
    # Actually, for 3x3 matrix multiplication, the rank is known to be 7 (Strassen 1969)
    # Let me use the explicit rank-7 decomposition
    
    # Standard Strassen for 3x3:
    # We partition the 3x3 matrices and use 7 products
    
    # Let's use a known rank-7 decomposition for 3x3 matrix multiplication
    # The decomposition can be represented as 7 triples (u, v, w) where
    # u is a 9-vector, v is a 9-vector, w is a 9-vector
    
    # Using the formulation from Strassen's paper:
    # For A*B where A, B are 3x3, C = A*B
    
    # Strassen's algorithm for 3x3 uses these 7 multiplications:
    # We'll construct the decomposition explicitly
    
    # Rank-7 decomposition for 3x3 matrix multiplication
    # Each rank-1 term is u_i * v_i * w_i where u_i, v_i, w_i are vectors of length 9
    
    # The 7 rank-1 components for 3x3:
    strassen_3x3 = [
        # (u, v, w) triples, each of length 9
        # u corresponds to A entries: a00,a01,a02,a10,a11,a12,a20,a21,a22
        # v corresponds to B entries: b00,b01,b02,b10,b11,b12,b20,b21,b22
        # w corresponds to C entries: c00,c01,c02,c10,c11,c12,c20,c21,c22
        
        # Term 1
        ([1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1], [1,0,0, 0,1,0, 0,0,1]),
        # Term 2
        ([0,1,0, 1,0,0, 0,0,0], [0,0,1, 1,0,0, 0,0,0], [0,1,0, 1,0,0, 0,0,0]),
        # Term 3
        ([0,0,1, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 1,0,0], [0,0,1, 0,0,0, 1,0,0]),
        # Term 4
        ([1,0,0, 0,0,0, 0,1,0], [0,0,1, 0,0,0, 0,0,1], [1,0,0, 0,0,0, 0,1,0]),
        # Term 5
        ([0,1,0, 0,0,0, 0,0,1], [0,0,0, 0,1,0, 0,0,1], [0,1,0, 0,0,0, 0,0,1]),
        # Term 6
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        # Term 7
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0]),
    ]
    
    # Wait, the above is not correct. Let me use a proper rank-7 decomposition.
    # The rank of 3x3 matrix multiplication is 7, but I need the exact decomposition.
    
    # Let me use a simpler approach: use schoolbook for all, but optimize the 3x3x1 part
    # Actually, let me just use the best known decomposition
    
    # For 3x3x7, the best known rank is 23 (using Strassen for the 3x3 blocks)
    # But I need the actual decomposition
    
    # Let me use a verified rank-7 decomposition for 3x3 matrix multiplication
    # From the literature, one such decomposition is:
    
    # Actually, let me just use schoolbook which gives rank 63, and try to improve
    # by using a better decomposition for the 3x3 blocks
    
    # For now, let's use the schoolbook decomposition (rank 63) as a baseline
    # and see if we can do better
    
    # Reset
    rank = 63
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    # First 3x3 block: i in 0..2, j in 0..2, k in 0..2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Second 3x3 block: i in 0..2, j in 0..2, k in 3..5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Third 3x1 block: i in 0..2, j in 0..2, k in 6
    for i in range(n):
        for j in range(m):
            k = 6
            U[i*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
