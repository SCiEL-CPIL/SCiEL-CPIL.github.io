"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-25 tensor decomposition for 2x2 * 2x7 matrix multiplication.
    Uses Strassen (rank 7) for three 2x2x2 blocks and schoolbook (rank 4) for the final 2x2x1 block.
    """
    import time
    start_time = time.time()
    DEADLINE = start_time + 120.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, 25)
    # V: (m*p, rank) = (14, 25)
    # W: (p*n, rank) = (14, 25)
    
    # Indices:
    # U index: i * m + j  -> 0..3 (i in 0..1, j in 0..1)
    # V index: j * p + k  -> 0..13 (j in 0..1, k in 0..6)
    # W index: k * n + i  -> 0..13 (k in 0..6, i in 0..1)
    
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_U(i, j, col, val):
        U[i * m + j][col] = val
        
    def set_V(j, k, col, val):
        V[j * p + k][col] = val
        
    def set_W(k, i, col, val):
        W[k * n + i][col] = val

    # Strassen decomposition for 2x2 * 2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_strassen_block(k_offset, col_offset):
        # k_offset: 0, 2, or 4 (for the first column of the 2-column block in B)
        # col_offset: 0, 7, 14
        
        c0 = col_offset
        
        # M1: (A00+A11)(B0k+B1k)
        # U: A00=1, A11=1
        set_U(0, 0, c0, 1)
        set_U(1, 1, c0, 1)
        # V: B0k=1, B1k=1
        set_V(0, k_offset, c0, 1)
        set_V(1, k_offset, c0, 1)
        # W: C00 += 1, C11 += 1
        set_W(k_offset, 0, c0, 1)
        set_W(k_offset+1, 1, c0, 1)

        # M2: (A10+A11)B0k
        set_U(1, 0, c0+1, 1)
        set_U(1, 1, c0+1, 1)
        set_V(0, k_offset, c0+1, 1)
        set_W(k_offset+1, 0, c0+1, 1) # C10 += M2

        # M3: A00(B1k-B0k)
        set_U(0, 0, c0+2, 1)
        set_V(1, k_offset, c0+2, 1)
        set_V(0, k_offset, c0+2, -1)
        set_W(k_offset, 1, c0+2, 1) # C01 += M3

        # M4: A11(B1k-B0k)
        set_U(1, 1, c0+3, 1)
        set_V(1, k_offset, c0+3, 1)
        set_V(0, k_offset, c0+3, -1)
        set_W(k_offset, 0, c0+3, 1) # C00 += M4
        set_W(k_offset+1, 0, c0+3, 1) # C10 += M4

        # M5: (A00+A01)B1k
        set_U(0, 0, c0+4, 1)
        set_U(0, 1, c0+4, 1)
        set_V(1, k_offset, c0+4, 1)
        set_W(k_offset, 0, c0+4, -1) # C00 -= M5
        set_W(k_offset, 1, c0+4, 1) # C01 += M5

        # M6: (A10-A00)(B0k+B1k)
        set_U(1, 0, c0+5, 1)
        set_U(0, 0, c0+5, -1)
        set_V(0, k_offset, c0+5, 1)
        set_V(1, k_offset, c0+5, 1)
        set_W(k_offset+1, 1, c0+5, 1) # C11 += M6

        # M7: (A01-A11)(B0k+B1k)
        set_U(0, 1, c0+6, 1)
        set_U(1, 1, c0+6, -1)
        set_V(0, k_offset, c0+6, 1)
        set_V(1, k_offset, c0+6, 1)
        set_W(k_offset, 0, c0+6, 1) # C00 += M7

    # Add 3 Strassen blocks for columns (0,1), (2,3), (4,5)
    add_strassen_block(0, 0)
    add_strassen_block(2, 7)
    add_strassen_block(4, 14)

    # Schoolbook decomposition for 2x2 * 2x1 (final column k=6)
    # C00 = A00*B06 + A01*B16
    # C10 = A10*B06 + A11*B16
    
    c0 = 21
    
    # M1: A00*B06
    set_U(0, 0, c0, 1)
    set_V(0, 6, c0, 1)
    set_W(6, 0, c0, 1)

    # M2: A01*B16
    set_U(0, 1, c0+1, 1)
    set_V(1, 6, c0+1, 1)
    set_W(6, 0, c0+1, 1)

    # M3: A10*B06
    set_U(1, 0, c0+2, 1)
    set_V(0, 6, c0+2, 1)
    set_W(6, 1, c0+2, 1)

    # M4: A11*B16
    set_U(1, 1, c0+3, 1)
    set_V(1, 6, c0+3, 1)
    set_W(6, 1, c0+3, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
