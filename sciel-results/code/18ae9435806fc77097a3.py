"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 33.
    # This is based on decomposing the 3x6 matrix into three 3x2 blocks.
    # For each 3x2 block, we use a rank-11 decomposition for 2x3x2 matrix multiplication.
    # The rank-11 decomposition is formed by:
    # 1. Strassen's algorithm (rank 7) for the 2x2x2 part (first 2 columns of B, first 2 rows of A).
    # 2. Schoolbook (rank 4) for the 2x1x2 part (last column of B, last row of A).
    
    # Total rank = 3 blocks * 11 terms/block = 33.
    rank = 33
    U = [[0] * rank for _ in range(n * m)] # 6 x 33
    V = [[0] * rank for _ in range(m * p)] # 18 x 33
    W = [[0] * rank for _ in range(p * n)] # 12 x 33
    
    # Helper to add a term
    # u: dict of (i,j) -> val for A (2x3)
    # v: dict of (j,k) -> val for B (3x6)
    # w: dict of (k,i) -> val for C (6x2) -> Note: W index is k*n + i
    def add_term(u_vals, v_vals, w_vals, col):
        for (i, j), val in u_vals.items():
            U[i * m + j][col] += val
        for (j, k), val in v_vals.items():
            V[j * p + k][col] += val
        for (k, i), val in w_vals.items():
            W[k * n + i][col] += val

    col = 0
    for block in range(3):
        # Block corresponds to columns 2*block to 2*block+1 of B, and C.
        # A is shared.
        # A indices: i in 0..1, j in 0..2.
        # B indices: j in 0..2, k in 2*block .. 2*block+1.
        # C indices: k in 2*block .. 2*block+1, i in 0..1.
        
        # Part 1: Strassen for A[:, 0:2] and B[0:2, 2*block:2*block+2]
        # A_sub is 2x2: A[i, j] for i in 0,1 and j in 0,1.
        # B_sub is 2x2: B[j, k] for j in 0,1 and k in 2*block, 2*block+1.
        
        # Strassen multipliers
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B10)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # We need to map these to U, V, W terms.
        # Each multiplier is a product of a linear form in A and a linear form in B.
        # The result contributes to C.
        
        # Let's define the linear forms for A and B for each multiplier.
        # A is 2x3. We only use cols 0,1.
        # B is 3x6. We only use rows 0,1 and cols 2*block, 2*block+1.
        
        # Multiplier 1: (A00+A11) * (B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0, 2*block): 1, (1, 2*block+1): 1}
        # C contribution:
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1?
        # Strassen formula:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So for M1:
        w1 = {(2*block, 0): 1, (2*block+1, 0): 1, (2*block, 1): 1, (2*block+1, 1): 1}
        # Wait, W index is k*n + i.
        # k is col of C (0..5), i is row of C (0..1).
        # C00 is k=2*block, i=0.
        # C01 is k=2*block+1, i=0.
        # C10 is k=2*block, i=1.
        # C11 is k=2*block+1, i=1.
        
        add_term(u1, v1, w1, col)
        col += 1
        
        # Multiplier 2: (A10+A11) * B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0, 2*block): 1}
        w2 = {(2*block, 1): 1} # C10 += M2
        add_term(u2, v2, w2, col)
        col += 1
        
        # Multiplier 3: A00 * (B11-B10)
        u3 = {(0,0): 1}
        v3 = {(1, 2*block+1): 1, (1, 2*block): -1}
        w3 = {(2*block+1, 0): 1, (2*block+1, 1): 1} # C01 += M3, C11 += M3
        add_term(u3, v3, w3, col)
        col += 1
        
        # Multiplier 4: A11 * (B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1, 2*block): 1, (0, 2*block): -1}
        w4 = {(2*block, 0): 1, (2*block, 1): 1} # C00 += M4, C10 += M4
        add_term(u4, v4, w4, col)
        col += 1
        
        # Multiplier 5: (A00+A01) * B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1, 2*block+1): 1}
        w5 = {(2*block, 0): -1, (2*block+1, 0): 1} # C00 -= M5, C01 += M5
        add_term(u5, v5, w5, col)
        col += 1
        
        # Multiplier 6: (A10-A00) * (B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0, 2*block): 1, (0, 2*block+1): 1}
        w6 = {(2*block+1, 1): 1} # C11 += M6
        add_term(u6, v6, w6, col)
        col += 1
        
        # Multiplier 7: (A01-A11) * (B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1, 2*block): 1, (1, 2*block+1): 1}
        w7 = {(2*block, 0): 1} # C00 += M7
        add_term(u7, v7, w7, col)
        col += 1
        
        # Part 2: Schoolbook for A[:, 2] and B[2, 2*block:2*block+2]
        # A_ext is 2x1: A[i, 2].
        # B_ext is 1x2: B[2, k].
        # C_ext is 2x2: C[i, k] += A[i, 2] * B[2, k].
        
        # Term 8: A02 * B20
        u8 = {(0,2): 1}
        v8 = {(2, 2*block): 1}
        w8 = {(2*block, 0): 1}
        add_term(u8, v8, w8, col)
        col += 1
        
        # Term 9: A02 * B21
        u9 = {(0,2): 1}
        v9 = {(2, 2*block+1): 1}
        w9 = {(2*block+1, 0): 1}
        add_term(u9, v9, w9, col)
        col += 1
        
        # Term 10: A12 * B20
        u10 = {(1,2): 1}
        v10 = {(2, 2*block): 1}
        w10 = {(2*block, 1): 1}
        add_term(u10, v10, w10, col)
        col += 1
        
        # Term 11: A12 * B21
        u11 = {(1,2): 1}
        v11 = {(2, 2*block+1): 1}
        w11 = {(2*block+1, 1): 1}
        add_term(u11, v11, w11, col)
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
