"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid Strassen (for 2x2 blocks) + schoolbook (for remaining columns)
    # n=2, m=5, p=6
    # Use Strassen for j in {0,1} and k in blocks of 2 (3 blocks of k: {0,1}, {2,3}, {4,5})
    # Schoolbook for j in {2,3,4}
    # Rank = 3 * 7 + 3 * 12 = 21 + 36 = 57
    #
    # Wait, let me reconsider. The schoolbook part for each j uses rank 12 (n*p = 2*6).
    # Strassen for each 2x2 block uses rank 7.
    # Total = 3*7 + 3*12 = 21 + 36 = 57.
    #
    # Actually, let me think more carefully. Can we do better?
    #
    # For the Strassen part: we have 2x2 matrices A (rows i=0,1; cols j=0,1) 
    # times 2x2 matrices B (rows j=0,1; cols k in each block of 2).
    # Strassen uses 7 multiplications per 2x2 block.
    #
    # For the schoolbook part: j in {2,3,4}, each column j of A times all of B.
    # This is essentially 1x5 times 5x6, but only using columns 2,3,4 of A.
    # For each j in {2,3,4}, we need rank n*p = 2*6 = 12.
    #
    # Total rank: 3*7 + 3*12 = 57.
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Strassen U coefficients: for each of the 7 products, what is the contribution to A elements
    # U[i*m+j, col] where i is row of A, j is col of A (within the 2x2 block)
    strassen_U = [
        # [i,j] coefficients for M1..M7
        # M1: (A00+A11)(B00+B11) -> A contributes A00=1, A11=1
        {0: 1, 1: 0, 2: 1, 3: 0},  # wait, let me use flat indexing
    ]
    
    # Let me use a cleaner approach. For the 2x2 block with i in {0,1}, j in {0,1} (local),
    # the 7 Strassen products have the following A-factor contributions:
    # M1: A00 + A11
    # M2: A10 + A11
    # M3: A00
    # M4: A11
    # M5: A00 + A01
    # M6: A10 - A00
    # M7: A01 - A11
    
    # So for local (i,j) in {0,1}^2, the U coefficients for each of 7 products:
    strassen_U_coeffs = [
        # M1: A00+A11
        {(0,0): 1, (1,1): 1},
        # M2: A10+A11
        {(1,0): 1, (1,1): 1},
        # M3: A00
        {(0,0): 1},
        # M4: A11
        {(1,1): 1},
        # M5: A00+A01
        {(0,0): 1, (0,1): 1},
        # M6: A10-A00
        {(1,0): 1, (0,0): -1},
        # M7: A01-A11
        {(0,1): 1, (1,1): -1},
    ]
    
    # B-factor contributions for 2x2 block with j_local in {0,1}, k_local in {0,1}:
    # M1: B00 + B11
    # M2: B00
    # M3: B10 - B11
    # M4: B01 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    strassen_V_coeffs = [
        # M1: B00+B11
        {(0,0): 1, (1,1): 1},
        # M2: B00
        {(0,0): 1},
        # M3: B10-B11
        {(1,0): 1, (1,1): -1},
        # M4: B01-B00
        {(0,1): 1, (0,0): -1},
        # M5: B11
        {(1,1): 1},
        # M6: B00+B01
        {(0,0): 1, (0,1): 1},
        # M7: B10+B11
        {(1,0): 1, (1,1): 1},
    ]
    
    # C-factor contributions (output 2x2):
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_W_coeffs = [
        # For each of 7 products, contribution to (k_local, i_local) in output
        # M1: C00 += 1, C11 += 1
        {(0,0): 1, (1,1): 1},
        # M2: C10 += 1, C11 -= 1
        {(1,0): 1, (1,1): -1},
        # M3: C01 += 1, C11 += 1
        {(0,1): 1, (1,1): 1},
        # M4: C00 += 1, C10 += 1
        {(0,0): 1, (1,0): 1},
        # M5: C00 -= 1, C01 += 1
        {(0,0): -1, (0,1): 1},
        # M6: C11 += 1
        {(1,1): 1},
        # M7: C00 += 1
        {(0,0): 1},
    ]
    
    # Total rank calculation:
    # 3 Strassen blocks (for k-blocks {0,1}, {2,3}, {4,5}), each with rank 7
    # 3 schoolbook columns (j=2,3,4), each with rank 12
    total_rank = 3 * 7 + 3 * 12  # 21 + 36 = 57
    
    rank = total_rank
    U = [[0] * rank for _ in range(n * m)]  # 10 x 57
    V = [[0] * rank for _ in range(m * p)]  # 30 x 57
    W = [[0] * rank for _ in range(p * n)]  # 12 x 57
    
    col = 0
    # Strassen blocks for j in {0,1}, k-blocks
    for k_block in range(3):
        k_base = k_block * 2  # 0, 2, 4
        for s in range(7):
            # U part: for local (i,j) in {0,1}^2, global i=i, j=j
            for (li, lj), coeff in strassen_U_coeffs[s].items():
                gi = li
                gj = lj  # j is 0 or 1 (first two columns)
                U[gi * m + gj][col] += coeff
            
            # V part: for local (lj, lk) in {0,1}^2, global j=lj, k=k_base+lk
            for (lj, lk), coeff in strassen_V_coeffs[s].items():
                gj = lj  # 0 or 1
                gk = k_base + lk
                V[gj * p + gk][col] += coeff
            
            # W part: for local (lk, li) in {0,1}^2, global k=k_base+lk, i=li
            for (lk, li), coeff in strassen_W_coeffs[s].items():
                gk = k_base + lk
                gi = li
                W[gk * n + gi][col] += coeff
            
            col += 1
    
    # Schoolbook part for j in {2,3,4}
    for j in range(2, 5):
        for k in range(p):
            for i in range(n):
                # One rank-1 term for (i, j, k)
                U[i * m + j][col] += 1
                V[j * p + k][col] += 1
                W[k * n + i][col] += 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
