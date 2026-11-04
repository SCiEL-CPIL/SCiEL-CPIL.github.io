"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # U: (n*m, rank) = (20, rank)
    # V: (m*p, rank) = (25, rank)
    # W: (p*n, rank) = (20, rank)
    
    # We'll build the decomposition in parts:
    # Part 1: [4,4,4] using Strassen (rank 49)
    # Part 2: [4,4,1] rank-1 (rank 1)
    # Part 3: [4,1,5] rank-1 (rank 1)
    # Total rank: 51
    
    rank = 51
    U = [[0.0] * rank for _ in range(n * m)]  # 20 x 51
    V = [[0.0] * rank for _ in range(m * p)]  # 25 x 51
    W = [[0.0] * rank for _ in range(p * n)]  # 20 x 51
    
    # Strassen [2,2,2] decomposition terms
    # For a 2x2 X * 2x2 Y -> 2x2 Z:
    # U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i
    strassen_terms = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U: [(i,j,coeff), ...], V: [(j,k,coeff), ...], W: [(k,i,coeff), ...]
        ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,1,1)]),   # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1)]),                      # M2
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1)]),                     # M3
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(1,0,1)]),                     # M4
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,1,1),(1,1,1)]),              # M5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)]),     # M6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1),(0,1,1)]),     # M7
    ]
    
    # For [4,4,4]: split i=i1+2*i2, j=j1+2*j2, k=k1+2*k2
    # 7 Strassen products at the 4x4 level:
    # Each product is a [2,2,2] tensor with specific U,V,W modifications
    
    # M1_4 = (A00+A11) * (B00+B11)
    # A00: i in {0,1}, j in {0,1} -> global i in {0,1}, j in {0,1}
    # A11: i in {2,3}, j in {2,3} -> global i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1} -> global j in {0,1}, k in {0,1}
    # B11: j in {2,3}, k in {2,3} -> global j in {2,3}, k in {2,3}
    
    # For each of the 7 Strassen terms, and for each of the 7 sub-terms:
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Let me define the 7 Strassen products at the 4x4 level:
    # Each has a U-part (linear combo of 2x2 blocks of A) and V-part (linear combo of 2x2 blocks of B)
    # and produces a 2x2 result that contributes to specific 2x2 blocks of C.
    
    # Strassen at 4x4 level:
    # P1 = (A00+A11) * (B00+B11) -> contributes to C00 and C11
    # P2 = (A10+A11) * B00        -> contributes to C10 and C11
    # P3 = A00 * (B01-B11)        -> contributes to C01 and C11
    # P4 = A11 * (B10-B00)        -> contributes to C10 and C11
    # P5 = (A00+A01) * B11        -> contributes to C01 and C11
    # P6 = (A10-A00) * (B00+B01)  -> contributes to C00, C01, C10, C11
    # P7 = (A01-A11) * (B10+B11)  -> contributes to C00, C01, C10, C11
    
    # Actually, let me just use the standard Strassen formulas:
    # C00 = P1 + P4 - P6 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 + P6
    
    # Where:
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # Each P is a 2x2 matrix = product of two 2x2 matrices.
    # Each P uses the [2,2,2] Strassen decomposition (7 terms).
    
    # For P = X * Y where X and Y are 2x2:
    # X is a linear combination of A-blocks, Y is a linear combination of B-blocks.
    # The [2,2,2] decomposition of X*Y gives 7 rank-1 terms.
    # Each rank-1 term has U (for X), V (for Y), W (for output Z).
    
    # The U part for X: X[i1,j1] = sum of coeff * A[i2*2+i1, j2*2+j1]
    # So in the global U matrix, for local (i1,j1) with coefficient c:
    #   U[(i2*2+i1)*5 + (j2*2+j1)] += c
    
    # Similarly for V and W.
    
    # Let me define the 7 Strassen products with their U and V block coefficients:
    # Each product t has:
    #   U_blocks: list of (i2, j2, coeff) for the A-blocks
    #   V_blocks: list of (j2, k2, coeff) for the B-blocks
    #   W_blocks: list of (k2, i2, coeff) for the C-blocks (from the [2,2,2] W)
    
    # Actually, it's cleaner to think of it as:
    # For product P_t = X_t * Y_t:
    #   X_t[i1,j1] = sum_{(i2,j2,c) in U_blocks_t} c * A[i2*2+i1, j2*2+j1]
    #   Y_t[j1,k1] = sum_{(j2,k2,c) in V_blocks_t} c * B[j2*2+j1, k2*2+k1]
    #   Z_t[k1,i1] = sum_{(k2,i2,c) in W_blocks_t} c * C[i2*2+i1, k2*2+k1]
    
    # And the [2,2,2] Strassen decomposition gives the local U,V,W for X*Y->Z.
    
    # Let me define the 7 products:
    products = [
        # (U_blocks, V_blocks)
        # U_blocks: [(i2, j2, coeff), ...]
        # V_blocks: [(j2, k2, coeff), ...]
        # P1 = (A00+A11)*(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # P2 = (A10+A11)*B00
        ([(1,0,1),(1,1,1)], [(0,0,1)]),
        # P3 = A00*(B01-B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),
        # P4 = A11*(B10-B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),
        # P5 = (A00+A01)*B11
        ([(0,0,1),(0,1,1)], [(1,1,1)]),
        # P6 = (A10-A00)*(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),
        # P7 = (A01-A11)*(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),
    ]
    
    # For each product t and each Strassen sub-term s:
    # The local U for the [2,2,2] tensor gives coefficients for X[i1,j1]
    # The local V gives coefficients for Y[j1,k1]
    # The local W gives coefficients for Z[k1,i1]
    
    # For sub-term s with local U coeffs u_s[(i1,j1)], local V coeffs v_s[(j1,k1)], local W coeffs w_s[(k1,i1)]:
    # For each (i2,j2,cU) in U_blocks_t and each (i1,j1,cu) in u_s:
    #   U[(i2*2+i1)*5 + (j2*2+j1)] += cU * cu
    # For each (j2,k2,cV) in V_blocks_t and each (j1,k1,cv) in v_s:
    #   V[(j2*2+j1)*5 + (k2*2+k1)] += cV * cv
    # For each (k2,i2,cW) in W_blocks_t... 
    
    # Wait, I need to figure out the W blocks. The output Z_t is a 2x2 matrix that contributes
    # to specific 2x2 blocks of C.
    
    # C00 = P1 + P4 - P6 + P7: Z from P1, P4, P6, P7 all go to C00 (i2=0,k2=0)
    # C01 = P3 + P5: Z from P3, P5 go to C01 (i2=0,k2=1)
    # C10 = P2 + P4: Z from P2, P4 go to C10 (i2=1,k2=0)
    # C11 = P1 - P3 + P5 + P6: Z from P1, P3, P5, P6 go to C11 (i2=1,k2=1)
    
    # So for product t, the W blocks tell us which C-blocks the output goes to:
    W_blocks = [
        # P1: C00 (+1) and C11 (+1)
        [(0,0,1),(1,1,1)],
        # P2: C10 (+1)
        [(1,0,1)],
        # P3: C01 (+1) and C11 (-1)
        [(0,1,1),(1,1,-1)],
        # P4: C10 (+1) and C11 (+1)
        [(1,0,1),(1,1,1)],
        # P5: C01 (+1) and C11 (+1)
        [(0,1,1),(1,1,1)],
        # P6: C00 (-1), C01... wait let me recheck.
        # C00 = P1 + P4 - P6 + P7: P6 contributes -1 to C00
        # C01 = P3 + P5: P6 doesn't contribute to C01
        # C10 = P2 + P4: P6 doesn't contribute to C10
        # C11 = P1 - P3 + P5 + P6: P6 contributes +1 to C11
        [(0,0,-1),(1,1,1)],
        # P7: C00 (+1), C11... 
        # C00 = P1 + P4 - P6 + P7: P7 contributes +1 to C00
        # C01 = P3 + P5: P7 doesn't contribute
        # C10 = P2 + P4: P7 doesn't contribute
        # C11 = P1 - P3 + P5 + P6: P7 doesn't contribute to C11
        # Wait, that doesn't seem right. Let me recheck Strassen.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M6 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M5 + M6
        # So P7 only contributes to C00 (C11 in 1-indexed).
        [(0,0,1)],
    ]
    
    # Now construct the decomposition:
    col = 0
    for t in range(7):
        U_blocks_t = products[t][0]
        V_blocks_t = products[t][1]
        W_blocks_t = W_blocks[t]
        
        for s in range(7):
            # Get the local Strassen term s
            u_local = strassen_terms[s][0]  # [(i1,j1,coeff), ...]
            v_local = strassen_terms[s][1]  # [(j1,k1,coeff), ...]
            w_local = strassen_terms[s][2]  # [(k1,i1,coeff), ...]
            
            # For U:
            for (i2, j2, cU) in U_blocks_t:
                for (i1, j1, cu) in u_local:
                    global_i = i2 * 2 + i1
                    global_j = j2 * 2 + j1
                    U[global_i * 5 + global_j][col] += cU * cu
            
            # For V:
            for (j2, k2, cV) in V_blocks_t:
                for (j1, k1, cv) in v_local:
                    global_j = j2 * 2 + j1
                    global_k = k2 * 2 + k1
                    V[global_j * 5 + global_k][col] += cV * cv
            
            # For W:
            for (k2, i2, cW) in W_blocks_t:
                for (k1, i1, cw) in w_local:
                    global_k = k2 * 2 + k1
                    global_i = i2 * 2 + i1
                    W[global_k * 4 + global_i][col] += cW * cw
            
            col += 1
    
    # col should be 49 now
    
    # Part 2: [4,4,1] - T[i,j,4] = 1 for i in 0..3, j in 0..3
    # Rank-1: U[i*5+j] = 1, V[j*5+4] = 1, W[4*4+i] = 1
    for i in range(4):
        for j in range(4):
            U[i * 5 + j][col] += 1
            V[j * 5 + 4][col] += 1
            W[4 * 4 + i][col] += 1
    col += 1
    
    # Part 3: [4,1,5] - T[i,4,k] = 1 for i in 0..3, k in 0..4
    # Rank-1: U[i*5+4] = 1, V[4*5+k] = 1, W[k*4+i] = 1
    for i in range(4):
        for k in range(5):
            U[i * 5 + 4][col] += 1
            V[4 * 5 + k][col] += 1
            W[k * 4 + i][col] += 1
    col += 1
    
    assert col == 51, f"Expected 51, got {col}"
    
    # Convert to integers where possible
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
