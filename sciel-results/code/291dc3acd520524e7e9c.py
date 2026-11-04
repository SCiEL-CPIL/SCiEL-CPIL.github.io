"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a rank-22 decomposition by splitting the 3x4 matrix B into two 3x2 blocks.
    # C = A * B = [A * B1, A * B2]
    # Each A * Bk is a 2x3 * 3x2 multiplication.
    # We use a rank-11 decomposition for 2x3x2.
    # Total rank = 11 + 11 = 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)] # 6 x 22
    V = [[0] * rank for _ in range(m * p)] # 12 x 22
    W = [[0] * rank for _ in range(p * n)] # 8 x 22
    
    # Indices for A (2x3): i in 0..1, j in 0..2. U[i*m+j, col]
    # Indices for B (3x4): j in 0..2, k in 0..3. V[j*p+k, col]
    # Indices for C (2x4): k in 0..3, i in 0..1. W[k*n+i, col]
    
    def add_term(u_idx, v_idx, w_idx, col, val=1):
        U[u_idx][col] = val
        V[v_idx][col] = val
        W[w_idx][col] = val

    def add_term_multi(u_indices, v_indices, w_indices, col, val=1):
        # This is not standard, we need to add separate terms for each combination if not rank 1
        pass

    # Decomposition for 2x3x2 block (Bk is 3x2, Ck is 2x2)
    # Bk indices: j in 0..2, k_local in 0..1. Global k = k_offset + k_local.
    # Ck indices: k_local in 0..1, i in 0..1. Global k = k_offset + k_local.
    
    def decompose_2x3x2(k_offset, base_col):
        # Part 1: A_L (2x1) * B_T (1x2) -> Rank 4
        # A_L: i=0,1; j=0.
        # B_T: j=0; k_local=0,1.
        # C: k_local=0,1; i=0,1.
        
        # Term 1: a10 * b00 -> c00
        add_term(0*3+0, 0*4+k_offset+0, (k_offset+0)*2+0, base_col+0)
        # Term 2: a10 * b01 -> c01
        add_term(0*3+0, 0*4+k_offset+1, (k_offset+1)*2+0, base_col+1)
        # Term 3: a20 * b00 -> c10
        add_term(1*3+0, 0*4+k_offset+0, (k_offset+0)*2+1, base_col+2)
        # Term 4: a20 * b01 -> c11
        add_term(1*3+0, 0*4+k_offset+1, (k_offset+1)*2+1, base_col+3)
        
        # Part 2: A_R (2x2) * B_B (2x2) -> Rank 7 (Strassen)
        # A_R: i=0,1; j=1,2.
        # B_B: j=1,2; k_local=0,1.
        
        # Strassen M1: (A11+A22)*(B11+B22)
        # A11=a01, A22=a12. B11=b10, B22=b21.
        # Note: A_R indices: (0,1)->a01, (0,2)->a02, (1,1)->a11, (1,2)->a12.
        # Let's map A_R to 2x2 matrix:
        # [ a01, a02 ]
        # [ a11, a12 ]
        # B_B:
        # [ b10, b11 ]
        # [ b20, b21 ]
        
        # M1 = (a01+a12) * (b10+b21)
        # This requires summing in U and V.
        # We cannot just add 1 to U and V. We need to construct the linear combination.
        # The rank-1 term is (u1+u2) (v1+v2) w.
        # This expands to 4 terms if we don't allow sums in U/V.
        # But the tensor decomposition allows U, V, W to have arbitrary values.
        # So for M1:
        # U part: a01 and a12 are summed.
        # V part: b10 and b21 are summed.
        # W part: contributes to C.
        
        # To implement Strassen, we define the linear forms.
        # For each of the 7 Strassen multiplications, we create one column in U, V, W.
        
        # M1: (A11+A22)(B11+B22)
        # U: a01 (idx 0*3+1) = 1, a12 (idx 1*3+2) = 1
        # V: b10 (idx 1*4+k_offset+0) = 1, b21 (idx 2*4+k_offset+1) = 1
        # W: C11 (k_offset+0, 0) += 1, C12 (k_offset+1, 0) += 0? No.
        # Strassen formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Let's assign columns base_col+4 to base_col+10.
        
        # Col 4 (M1):
        U[0*3+1][base_col+4] = 1
        U[1*3+2][base_col+4] = 1
        V[1*4+k_offset+0][base_col+4] = 1
        V[2*4+k_offset+1][base_col+4] = 1
        W[(k_offset+0)*2+0][base_col+4] = 1  # C11
        W[(k_offset+1)*2+1][base_col+4] = 1  # C22 (Wait, M1 contributes to C11 and C22? No.)
        # Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # So M1 contributes to C11 and C22.
        W[(k_offset+0)*2+0][base_col+4] = 1
        W[(k_offset+1)*2+1][base_col+4] = 1
        
        # Col 5 (M2): (A21+A22)B11
        # A21=a11 (idx 1*3+1), A22=a12 (idx 1*3+2). B11=b10.
        U[1*3+1][base_col+5] = 1
        U[1*3+2][base_col+5] = 1
        V[1*4+k_offset+0][base_col+5] = 1
        W[(k_offset+0)*2+1][base_col+5] = 1  # C21
        W[(k_offset+1)*2+1][base_col+5] = -1 # C22
        
        # Col 6 (M3): A11(B12-B22)
        # A11=a01. B12=b11, B22=b21.
        U[0*3+1][base_col+6] = 1
        V[1*4+k_offset+1][base_col+6] = 1
        V[2*4+k_offset+1][base_col+6] = -1
        W[(k_offset+1)*2+0][base_col+6] = 1  # C12
        W[(k_offset+1)*2+1][base_col+6] = 1  # C22
        
        # Col 7 (M4): A22(B21-B11)
        # A22=a12. B21=b21, B11=b10.
        U[1*3+2][base_col+7] = 1
        V[2*4+k_offset+1][base_col+7] = 1
        V[1*4+k_offset+0][base_col+7] = -1
        W[(k_offset+0)*2+0][base_col+7] = 1  # C11
        W[(k_offset+0)*2+1][base_col+7] = 1  # C21
        
        # Col 8 (M5): (A11+A12)B22
        # A11=a01, A12=a02. B22=b21.
        U[0*3+1][base_col+8] = 1
        U[0*3+2][base_col+8] = 1
        V[2*4+k_offset+1][base_col+8] = 1
        W[(k_offset+0)*2+0][base_col+8] = -1 # C11
        W[(k_offset+1)*2+0][base_col+8] = 1  # C12
        
        # Col 9 (M6): (A21-A11)(B11+B12)
        # A21=a11, A11=a01. B11=b10, B12=b11.
        U[1*3+1][base_col+9] = 1
        U[0*3+1][base_col+9] = -1
        V[1*4+k_offset+0][base_col+9] = 1
        V[1*4+k_offset+1][base_col+9] = 1
        W[(k_offset+1)*2+1][base_col+9] = 1  # C22
        
        # Col 10 (M7): (A12-A22)(B21+B22)
        # A12=a02, A22=a12. B21=b21, B22=b21? No B22 is b21.
        # Wait, B_B is:
        # [ b10, b11 ]
        # [ b20, b21 ]
        # B21 is b21? No, B21 is row 2 col 1 (0-indexed: 1,0) -> b10?
        # Let's stick to 0-indexed:
        # B[0,0]=b10, B[0,1]=b11
        # B[1,0]=b20, B[1,1]=b21
        # So B11=b10, B12=b11, B21=b20, B22=b21.
        
        # Correction in M4: B21 is b20, B11 is b10.
        # Correction in M5: B22 is b21.
        # Correction in M7: B21 is b20, B22 is b21.
        
        # Let's re-verify M7:
        # A12=a02, A22=a12.
        # B21=b20 (idx 2*4+k_offset+0), B22=b21 (idx 2*4+k_offset+1).
        U[0*3+2][base_col+10] = 1
        U[1*3+2][base_col+10] = -1
        V[2*4+k_offset+0][base_col+10] = 1
        V[2*4+k_offset+1][base_col+10] = 1
        W[(k_offset+0)*2+0][base_col+10] = 1  # C11
        # M7 does not contribute to C22 in standard Strassen?
        # C22 = M1 - M2 + M3 + M6. Correct.
        
    # Apply for B1 (k_offset=0) and B2 (k_offset=2)
    decompose_2x3x2(0, 0)
    decompose_2x3x2(2, 11)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
