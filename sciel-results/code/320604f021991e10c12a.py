"""Matrix multiplication 2,2,4 - Strassen rank-14 decomposition."""
DIMENSIONS = [2, 2, 4]

def construct():
    """
    Strassen-based rank-14 decomposition for 2x2 * 2x4 matrix multiplication.
    
    A (2x2) * B (2x4) = C (2x4)
    Split B into B1 (2x2) and B2 (2x2), apply Strassen to each.
    
    Trace convention: W[k*n+i, r] where n=2, so W index = k*2+i
    U index: i*m+j = i*2+j (i row of A, j col of A)
    V index: j*p+k = j*4+k (j row of B, k col of B)
    """
    n, m, p = 2, 2, 4  # A is n x m, B is m x p, C is n x p
    
    # Rank-14: 7 Strassen products for B1 (cols 0-1), 7 for B2 (cols 2-3)
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # Strassen indices for a 2x2 * 2x2 block:
    # A = [[A11, A12], [A21, A22]] -> U indices: A11=0, A12=1, A21=2, A22=3
    # B = [[B11, B12], [B21, B22]] -> V indices depend on which block
    
    # For block 1 (B cols 0,1): B11=V[0], B12=V[1], B21=V[4], B22=V[5]
    # For block 2 (B cols 2,3): B11=V[2], B12=V[3], B21=V[6], B22=V[7]
    
    def add_strassen_block(b_offset, w_offset, col_start):
        """
        Add 7 Strassen rank-1 terms for one 2x2*2x2 block.
        b_offset: 0 for B1 (cols 0-1), 2 for B2 (cols 2-3)
        w_offset: 0 for C cols 0-1, 2 for C cols 2-3
        col_start: starting column in U,V,W
        """
        # B indices: B11 = b_offset+0, B12 = b_offset+1, B21 = b_offset+4, B22 = b_offset+5
        B11 = b_offset + 0
        B12 = b_offset + 1
        B21 = b_offset + 4
        B22 = b_offset + 5
        
        # C indices: C11 = w_offset+0, C12 = w_offset+1, C21 = w_offset+2, C22 = w_offset+3
        # W index = k*n+i, so C[i][k] -> W[k*2+i]
        # C11 (i=0,k=0) -> W[0], C12 (i=0,k=1) -> W[1]
        # C21 (i=1,k=0) -> W[2], C22 (i=1,k=1) -> W[3]
        C11 = w_offset + 0
        C12 = w_offset + 1
        C21 = w_offset + 2
        C22 = w_offset + 3
        
        # A indices: A11=0, A12=1, A21=2, A22=3
        A11, A12, A21, A22 = 0, 1, 2, 3
        
        # P1 = A11 * B11
        c = col_start + 0
        U[A11][c] = 1
        V[B11][c] = 1
        W[C11][c] = 1
        W[C21][c] = 1  # P1 contributes to both C11 and C21? No...
        
        # Let me redo this properly with the Strassen formulas:
        # P1 = A11 * B11
        # P2 = A22 * B22
        # P3 = (A11+A22) * (B11+B22)
        # P4 = (A21+A22) * B22
        # P5 = A11 * (B12-B22)
        # P6 = (A21-A11) * B11
        # P7 = (A11+A12) * B21
        #
        # C11 = P1 + P3 - P2 + P4
        # C12 = P5 + P2
        # C21 = P6 + P1
        # C22 = P2 - P3 + P5 + P7
        
        # Reset and redo
        U[A11][c] = 0
        U[A12][c] = 0
        U[A21][c] = 0
        U[A22][c] = 0
        for row in range(8):
            V[row][c] = 0
        for row in range(8):
            W[row][c] = 0
        
        # P1 = A11 * B11
        c = col_start + 0
        U[A11][c] = 1
        V[B11][c] = 1
        # C11 += P1, C21 += P1
        W[C11][c] = 1
        W[C21][c] = 1
        
        # P2 = A22 * B22
        c = col_start + 1
        U[A22][c] = 1
        V[B22][c] = 1
        # C11 -= P2, C12 += P2, C22 += P2
        W[C11][c] = -1
        W[C12][c] = 1
        W[C22][c] = 1
        
        # P3 = (A11+A22) * (B11+B22)
        c = col_start + 2
        U[A11][c] = 1
        U[A22][c] = 1
        V[B11][c] = 1
        V[B22][c] = 1
        # C11 += P3, C22 -= P3
        W[C11][c] = 1
        W[C22][c] = -1
        
        # P4 = (A21+A22) * B22
        c = col_start + 3
        U[A21][c] = 1
        U[A22][c] = 1
        V[B22][c] = 1
        # C11 += P4
        W[C11][c] = 1
        
        # P5 = A11 * (B12-B22)
        c = col_start + 4
        U[A11][c] = 1
        V[B12][c] = 1
        V[B22][c] = -1
        # C12 += P5, C22 += P5
        W[C12][c] = 1
        W[C22][c] = 1
        
        # P6 = (A21-A11) * B11
        c = col_start + 5
        U[A21][c] = 1
        U[A11][c] = -1
        V[B11][c] = 1
        # C21 += P6
        W[C21][c] = 1
        
        # P7 = (A11+A12) * B21
        c = col_start + 6
        U[A11][c] = 1
        U[A12][c] = 1
        V[B21][c] = 1
        # C22 += P7
        W[C22][c] = 1
    
    # Block 1: B cols 0-1, C cols 0-1
    add_strassen_block(0, 0, 0)
    # Block 2: B cols 2-3, C cols 2-3
    add_strassen_block(2, 2, 7)
    
    return {"U": U, "V": V, "W": W}
