"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 14 decomposition using two independent Strassen algorithms for the two 2x2 blocks of B.
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [i, j]]
    # C = [[ae+bi, af+bj], [ce+di, cf+dj]]
    # M1 = (a+d)(e+i)
    # M2 = (c+d)(e)
    # M3 = (a)(j-i)
    # M4 = (d)(f-e)
    # M5 = (a+c)(j)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(i+j)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2): 0:a, 1:b, 2:c, 3:d
    # Indices for B (2x2): 0:e, 1:f, 2:i, 3:j
    
    def add_strassen(offset_u, offset_v, offset_w, b_start_col):
        # b_start_col is 0 for left block, 2 for right block
        # V indices: j*p + k. j in {0,1}, k in {b_start_col, b_start_col+1}
        # For j=0, k=b_start_col -> index 0*4 + b_start_col = b_start_col
        # For j=0, k=b_start_col+1 -> index b_start_col+1
        # For j=1, k=b_start_col -> index 1*4 + b_start_col = 4 + b_start_col
        # For j=1, k=b_start_col+1 -> index 4 + b_start_col + 1
        
        # U indices: i*m + j. i in {0,1}, j in {0,1}
        # 0:a, 1:b, 2:c, 3:d
        
        # W indices: k*n + i. k in {b_start_col, b_start_col+1}, i in {0,1}
        # k=b_start_col, i=0 -> b_start_col*2 + 0 = 2*b_start_col
        # k=b_start_col, i=1 -> 2*b_start_col + 1
        # k=b_start_col+1, i=0 -> 2*(b_start_col+1)
        # k=b_start_col+1, i=1 -> 2*(b_start_col+1) + 1

        # Term 1: (a+d)(e+i)
        # U: a(0)+d(3) -> U[0, off]=1, U[3, off]=1
        # V: e(0)+i(2) -> V[b_start_col, off]=1, V[4+b_start_col, off]=1
        # W: contributes to C00 and C11?
        # C00 = M1 - M4 + M5 + M7. So W for C00 (idx 2*b_start_col) gets 1.
        # C11 = M1 - M2 + M3 + M6. So W for C11 (idx 2*b_start_col+1) gets 1.
        
        U[0][offset_u] = 1
        U[3][offset_u] = 1
        V[b_start_col][offset_u] = 1
        V[4 + b_start_col][offset_u] = 1
        W[2 * b_start_col][offset_u] = 1
        W[2 * b_start_col + 1][offset_u] = 1
        
        # Term 2: (c+d)(e)
        U[2][offset_u+1] = 1
        U[3][offset_u+1] = 1
        V[b_start_col][offset_u+1] = 1
        # C10 = M2 + M4. W for C10 (idx 2*b_start_col+1) gets 1.
        # Wait, C10 is row 1, col 0. Index in W is k*n+i.
        # k=b_start_col, i=1 -> 2*b_start_col + 1.
        W[2 * b_start_col + 1][offset_u+1] = 1
        
        # Term 3: (a)(j-i)
        U[0][offset_u+2] = 1
        V[b_start_col+1][offset_u+2] = 1 # j is at col b_start_col+1, row 0
        V[4 + b_start_col+1][offset_u+2] = -1 # i is at col b_start_col+1, row 1
        # C01 = M3 + M4. W for C01 (idx 2*b_start_col+1) gets 1.
        # Wait, C01 is row 0, col 1. Index in W is k*n+i.
        # k=b_start_col+1, i=0 -> 2*(b_start_col+1) + 0 = 2*b_start_col+2.
        W[2 * b_start_col + 2][offset_u+2] = 1

        # Term 4: (d)(f-e)
        U[3][offset_u+3] = 1
        V[b_start_col+1][offset_u+3] = 1 # f
        V[4 + b_start_col][offset_u+3] = -1 # e
        # C00 = ... - M4 ... W for C00 (idx 2*b_start_col) gets -1.
        # C01 = ... + M4 ... W for C01 (idx 2*b_start_col+2) gets 1.
        # C10 = ... + M4 ... W for C10 (idx 2*b_start_col+1) gets 1.
        # C11 = ... (no M4)
        W[2 * b_start_col][offset_u+3] = -1
        W[2 * b_start_col + 1][offset_u+3] = 1
        W[2 * b_start_col + 2][offset_u+3] = 1

        # Term 5: (a+c)(j)
        U[0][offset_u+4] = 1
        U[2][offset_u+4] = 1
        V[4 + b_start_col+1][offset_u+4] = 1 # j
        # C00 = ... + M5 ... W for C00 (idx 2*b_start_col) gets 1.
        W[2 * b_start_col][offset_u+4] = 1

        # Term 6: (c-a)(e+f)
        U[2][offset_u+5] = 1
        U[0][offset_u+5] = -1
        V[b_start_col][offset_u+5] = 1 # e
        V[b_start_col+1][offset_u+5] = 1 # f
        # C11 = ... + M6 ... W for C11 (idx 2*b_start_col+1) gets 1.
        W[2 * b_start_col + 1][offset_u+5] = 1

        # Term 7: (b-d)(i+j)
        U[1][offset_u+6] = 1
        U[3][offset_u+6] = -1
        V[4 + b_start_col][offset_u+6] = 1 # i
        V[4 + b_start_col+1][offset_u+6] = 1 # j
        # C00 = ... + M7 ... W for C00 (idx 2*b_start_col) gets 1.
        W[2 * b_start_col][offset_u+6] = 1

    # Left block (cols 0, 1)
    add_strassen(0, 0, 0, 0)
    # Right block (cols 2, 3)
    add_strassen(7, 0, 0, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
