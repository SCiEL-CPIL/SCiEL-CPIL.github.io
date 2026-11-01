"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # M_{2,3,6} = M_{2,2,6} + M_{2,1,6}
    # M_{2,2,6} = 3 * M_{2,2,2} (Strassen) -> Rank 21
    # M_{2,1,6} = Schoolbook -> Rank 12
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: M_{2,2,6}
    # A is 2x2 (cols 0,1 of A_full)
    # B is 2x6 (rows 0,1 of B_full)
    # C is 2x6
    
    for block in range(3):
        # B block is cols 2*block to 2*block+1
        b_col_start = 2 * block
        
        # Strassen for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        U[0][col] = 1; U[3][col] = 1
        V[0][col] = 1; V[2][col] = 1
        W[0][col] = 1; W[5][col] = 1 # C00, C11? No, W index is k*n+i.
        # C is 2x2.
        # C00 is W[0*2+0]=W[0]. C11 is W[1*2+1]=W[3].
        # Wait, W shape is p*n. For 2x2 block, p=2, n=2.
        # Indices: k in [0,1], i in [0,1].
        # C00: k=0, i=0 -> W[0].
        # C01: k=1, i=0 -> W[1].
        # C10: k=0, i=1 -> W[2].
        # C11: k=1, i=1 -> W[3].
        
        # Correction for W indices in the full matrix
        # Full C is 2x6.
        # Block b contributes to C[:, 2*b : 2*b+2].
        # So C00 of block b is C[0, 2*b] -> W[ (2*b)*2 + 0 ] = W[4*b].
        # C01 of block b is C[0, 2*b+1] -> W[ (2*b+1)*2 + 0 ] = W[4*b+2].
        # C10 of block b is C[1, 2*b] -> W[ (2*b)*2 + 1 ] = W[4*b+1].
        # C11 of block b is C[1, 2*b+1] -> W[ (2*b+1)*2 + 1 ] = W[4*b+3].
        
        # Let's define helper to get W index
        def w_idx(k, i):
            return k * n + i
        
        # M1: (A00+A11)(B00+B11)
        # U: A00 (0,0), A11 (1,1)
        U[0][col] = 1; U[1*2+1][col] = 1 # A is 2x3. A11 is row 1, col 1. Index 1*3+1=4?
        # A index: i*m + j.
        # A00: 0. A11: 1*3+1=4.
        # V: B00 (0,0), B11 (1,1). B is 3x6.
        # B00: 0. B11: 1*6 + (2*b+1).
        V[0][col] = 1; V[1*6 + (2*b+1)][col] = 1
        # W: C00, C11 of block.
        W[w_idx(2*b, 0)][col] = 1
        W[w_idx(2*b+1, 1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        U[1*3+0][col] = 1; U[1*3+1][col] = 1
        V[0][col] = 1
        W[w_idx(2*b, 1)][col] = 1 # C10
        W[w_idx(2*b+1, 1)][col] = 1 # C11
        col += 1
        
        # M3: A00(B01-B11)
        U[0][col] = 1
        V[1*6 + (2*b+1)][col] = -1; V[1*6 + (2*b)][col] = 1 # B01 is (0, 2*b+1), B11 is (1, 2*b+1)?
        # B is 3x6.
        # B_T is 2x2. Rows 0,1. Cols 2*b, 2*b+1.
        # B00: (0, 2*b). B01: (0, 2*b+1).
        # B10: (1, 2*b). B11: (1, 2*b+1).
        V[0*6 + (2*b+1)][col] = 1; V[1*6 + (2*b+1)][col] = -1
        W[w_idx(2*b+1, 0)][col] = 1 # C01
        W[w_idx(2*b+1, 1)][col] = -1 # C11
        col += 1
        
        # M4: A11(B10-B00)
        U[1*3+1][col] = 1
        V[1*6 + (2*b)][col] = 1; V[0*6 + (2*b)][col] = -1
        W[w_idx(2*b, 1)][col] = 1 # C10
        W[w_idx(2*b+1, 1)][col] = -1 # C11
        col += 1
        
        # M5: (A00+A01)B11
        U[0][col] = 1; U[0*3+1][col] = 1
        V[1*6 + (2*b+1)][col] = 1
        W[w_idx(2*b+1, 0)][col] = 1 # C01
        W[w_idx(2*b+1, 1)][col] = 1 # C11
        col += 1
        
        # M6: (A10-A00)(B00+B01)
        U[1*3+0][col] = 1; U[0][col] = -1
        V[0*6 + (2*b)][col] = 1; V[0*6 + (2*b+1)][col] = 1
        W[w_idx(2*b, 0)][col] = 1 # C00
        W[w_idx(2*b+1, 0)][col] = 1 # C01
        col += 1
        
        # M7: (A01-A11)(B10+B11)
        U[0*3+1][col] = 1; U[1*3+1][col] = -1
        V[1*6 + (2*b)][col] = 1; V[1*6 + (2*b+1)][col] = 1
        W[w_idx(2*b, 0)][col] = -1 # C00
        W[w_idx(2*b, 1)][col] = 1 # C10
        col += 1

    # Part 2: M_{2,1,6}
    # A is 2x1 (col 2 of A_full)
    # B is 1x6 (row 2 of B_full)
    # C is 2x6
    
    for k in range(6):
        for i in range(2):
            # Term A_i2 * B_2k
            U[i*3+2][col] = 1
            V[2*6+k][col] = 1
            W[w_idx(k, i)][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
