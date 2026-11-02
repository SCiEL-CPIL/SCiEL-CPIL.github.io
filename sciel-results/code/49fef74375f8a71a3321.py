"""Program for matrix multiplication 3,3,6 with rank 27 decomposition."""
DIMENSIONS = [3, 3, 6]

def construct():
    """
    Construct a rank-27 decomposition for multiplying a 3x3 matrix A by a 3x6 matrix B.
    
    The key insight is that B can be viewed as two 3x3 blocks B1 and B2.
    We can reuse the same 3x3x3 schoolbook decomposition for both blocks,
    sharing the U factors (which depend only on A) but with different V and W factors
    for each block.
    
    U shape: (n*m, rank) = (9, 27)
    V shape: (m*p, rank) = (18, 27)  
    W shape: (p*n, rank) = (18, 27)
    """
    n, m, p = DIMENSIONS
    rank = 27
    
    # Initialize factor matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each of the 27 terms in the 3x3x3 decomposition:
    # i, j, k each range over 0, 1, 2
    # U[3*i + j, term] = 1
    # For block 1 (columns 0-2 of B): V[3*j + k, term] = 1, W[3*k + i, term] = 1
    # For block 2 (columns 3-5 of B): V[3*j + (k+3), term] = 1, W[3*(k+3) + i, term] = 1
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                term_idx = i * 9 + j * 3 + k
                
                # U factor: depends on i, j
                U[3 * i + j][term_idx] = 1.0
                
                # V factor for block 1 (B1, columns 0-2)
                V[3 * j + k][term_idx] = 1.0
                
                # V factor for block 2 (B2, columns 3-5)
                V[3 * j + (k + 3)][term_idx] = 1.0
                
                # W factor for block 1 (C1, columns 0-2)
                W[3 * k + i][term_idx] = 1.0
                
                # W factor for block 2 (C2, columns 3-5)
                W[3 * (k + 3) + i][term_idx] = 1.0
    
    return {"U": U, "V": V, "W": W}
