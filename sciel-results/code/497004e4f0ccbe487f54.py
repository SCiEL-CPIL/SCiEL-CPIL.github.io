"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 7
    # We will build the decomposition in parts.
    # Part 1: 4x4x4 multiplication using 8 blocks of 2x2x2 (Rank 7 each) -> Rank 56
    # Part 2: 4x4x3 multiplication using 4 blocks of 2x2x2 (for k=0,1) and 4 blocks of 2x2x1 (for k=2) -> Rank 28 + 16 = 44
    # Total Rank: 100
    
    rank = 100
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_222_block(i0, j0, k0, start_col):
        # i0, j0, k0 are the starting indices for the 2x2x2 block
        # Local indices: ii in 0,1; jj in 0,1; kk in 0,1
        # Global i = i0 + ii, j = j0 + jj, k = k0 + kk
        
        # M1 = (A00+A11)(B00+B11)
        U[i0*4 + j0][start_col] = 1
        U[(i0+1)*4 + (j0+1)][start_col] = 1
        V[j0*7 + k0][start_col] = 1
        V[(j0+1)*7 + (k0+1)][start_col] = 1
        W[k0*4 + i0][start_col] = 1
        W[(k0+1)*4 + (i0+1)][start_col] = 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*4 + j0][start_col+1] = 1
        U[(i0+1)*4 + (j0+1)][start_col+1] = 1
        V[j0*7 + k0][start_col+1] = 1
        W[k0*4 + (i0+1)][start_col+1] = 1
        
        # M3 = A00(B01-B11)
        U[i0*4 + j0][start_col+2] = 1
        V[j0*7 + (k0+1)][start_col+2] = 1
        V[(j0+1)*7 + (k0+1)][start_col+2] = -1
        W[(k0+1)*4 + i0][start_col+2] = 1
        
        # M4 = A11(B10-B00)
        U[(i0+1)*4 + (j0+1)][start_col+3] = 1
        V[(j0+1)*7 + k0][start_col+3] = 1
        V[j0*7 + k0][start_col+3] = -1
        W[k0*4 + (i0+1)][start_col+3] = 1
        
        # M5 = (A00+A01)B11
        U[i0*4 + j0][start_col+4] = 1
        U[i0*4 + (j0+1)][start_col+4] = 1
        V[(j0+1)*7 + (k0+1)][start_col+4] = 1
        W[(k0+1)*4 + i0][start_col+4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i0+1)*4 + j0][start_col+5] = 1
        U[i0*4 + j0][start_col+5] = -1
        V[j0*7 + k0][start_col+5] = 1
        V[j0*7 + (k0+1)][start_col+5] = 1
        W[k0*4 + (i0+1)][start_col+5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i0*4 + (j0+1)][start_col+6] = 1
        U[(i0+1)*4 + (j0+1)][start_col+6] = -1
        V[(j0+1)*7 + k0][start_col+6] = 1
        V[(j0+1)*7 + (k0+1)][start_col+6] = 1
        W[(k0+1)*4 + i0][start_col+6] = 1
        
        return start_col + 7

    # Helper to add a 2x2x1 block
    def add_221_block(i0, j0, k0, start_col):
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # 4 multiplications
        
        # M1 = A00 B0
        U[i0*4 + j0][start_col] = 1
        V[j0*7 + k0][start_col] = 1
        W[k0*4 + i0][start_col] = 1
        
        # M2 = A01 B1
        U[i0*4 + (j0+1)][start_col+1] = 1
        V[(j0+1)*7 + k0][start_col+1] = 1
        W[k0*4 + i0][start_col+1] = 1
        
        # M3 = A10 B0
        U[(i0+1)*4 + j0][start_col+2] = 1
        V[j0*7 + k0][start_col+2] = 1
        W[k0*4 + (i0+1)][start_col+2] = 1
        
        # M4 = A11 B1
        U[(i0+1)*4 + (j0+1)][start_col+3] = 1
        V[(j0+1)*7 + k0][start_col+3] = 1
        W[k0*4 + (i0+1)][start_col+3] = 1
        
        return start_col + 4

    # Part 1: 4x4x4 (k=0..3)
    # 8 blocks of 2x2x2
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            for k0 in [0, 2]:
                col = add_222_block(i0, j0, k0, col)
                
    # Part 2: 4x4x3 (k=4..6)
    # Wait, the problem says B is 4x7. So k goes from 0 to 6.
    # My "Part 1" used k=0..3.
    # My "Part 2" should use k=4..6.
    # For k=4,5: 2x2x2 blocks.
    # For k=6: 2x2x1 blocks.
    
    # k=4,5 -> 2x2x2 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            col = add_222_block(i0, j0, 4, col) # k0=4, covers k=4,5
            
    # k=6 -> 2x2x1 blocks
    for i0 in [0, 2]:
        for j0 in [0, 2]:
            col = add_221_block(i0, j0, 6, col) # k0=6, covers k=6
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
