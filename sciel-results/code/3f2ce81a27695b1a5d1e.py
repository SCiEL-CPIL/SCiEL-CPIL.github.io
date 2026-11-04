"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block-Strassen decomposition of rank 189.
    # Treat 6x6 as 3x3 blocks of 2x2 matrices.
    # Rank = R(3x3 schoolbook) * R(2x2 Strassen) = 27 * 7 = 189.
    n, m, p = DIMENSIONS
    block_size = 2
    num_blocks = n // block_size # 3
    
    # Strassen 2x2 decomposition (rank 7)
    # U_s, V_s, W_s for 2x2 multiplication
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # c in {0..6}
    
    # Strassen terms:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # We construct the full 6x6 decomposition by iterating over the 3x3 block structure
    # and for each block multiplication, using the Strassen decomposition.
    
    # Total rank = 3*3*7 = 63? No.
    # The 3x3 block multiplication is schoolbook: C_ij = sum_k A_ik B_kj.
    # Each A_ik, B_kj, C_ij is a 2x2 matrix.
    # So for each (i,j,k) in 3x3x3, we perform a 2x2 matrix multiplication.
    # Rank for one 2x2 multiplication is 7.
    # Total rank = 27 * 7 = 189.
    
    rank = (n // block_size) ** 3 * 7
    # n=6, block_size=2, num_blocks=3. rank = 27 * 7 = 189.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # U_s[2*2, 7], V_s[2*2, 7], W_s[2*2, 7]
    # Row index in U_s: 2*i + j (row i, col j of A)
    # Row index in V_s: 2*j + k (row j, col k of B)
    # Row index in W_s: 2*k + i (row k, col i of C)
    
    # Strassen terms mapping:
    # c=0: (A00+A11)(B00+B11) -> A00, A11 ; B00, B11 ; C00+=, C01+=, C10+=, C11+=
    # c=1: (A10+A11)B00 -> A10, A11 ; B00 ; C10+=, C11+=
    # c=2: A00(B01-B11) -> A00 ; B01, B11 ; C01+=, C11-=
    # c=3: A11(B10-B00) -> A11 ; B10, B00 ; C10+=, C00-=
    # c=4: (A00+A01)B11 -> A00, A01 ; B11 ; C01+=, C11+=
    # c=5: (A10-A00)(B00+B01) -> A10, A00 ; B00, B01 ; C10+=, C11+=, C00-=, C01-=
    # c=6: (A01-A11)(B10+B11) -> A01, A11 ; B10, B11 ; C01+=, C11+=, C10-=, C00-=
    
    # Let's define the Strassen factor matrices explicitly.
    # U_s[i, c], V_s[j, c], W_s[k, c]
    # i,j,k in 0..1
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 1, 0, 0, -1], # A10
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [1, 0, 1, 0, 1, 1, 1],  # B10
        [0, 0, -1, 0, 0, 1, 0], # B01
        [1, 0, 1, 0, 1, 0, 1]   # B11
    ]
    
    W_s = [
        [1, 0, 0, -1, 0, -1, -1], # C00
        [0, 0, 0, 0, 1, 1, 0],    # C10
        [0, 1, 1, 0, 1, -1, 1],   # C01
        [1, 1, -1, 0, 1, 1, 1]    # C11
    ]
    
    # Verify Strassen:
    # C00 = M0 - M3 - M5 - M6
    # M0 = (A00+A11)(B00+B11)
    # M3 = A11(B10-B00)
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # This is getting error-prone to verify manually.
    # I will use a standard known correct Strassen implementation.
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 - M5 + M6
    
    # Using 0-indexed:
    # C00 = M0 - M3 - M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 - M4 + M5
    
    # Let's re-map:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M3 - M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M2 - M4 + M5
    
    # Let's build U_s, V_s, W_s based on this.
    # c=0 (M0): A00=1, A11=1; B00=1, B11=1; C00=1, C11=1
    # c=1 (M1): A10=1, A11=1; B00=1; C10=1
    # c=2 (M2): A00=1; B01=1, B11=-1; C01=1
    # c=3 (M3): A11=1; B10=1, B00=-1; C00=-1, C10=1
    # c=4 (M4): A00=1, A01=1; B11=1; C01=1, C11=-1
    # c=5 (M5): A10=1, A00=-1; B00=1, B01=1; C00=-1, C11=1
    # c=6 (M6): A01=1, A11=-1; B10=1, B11=1; C00=1, C11=-1
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 0, 1],  # B10
        [0, 0, 1, 0, 0, 1, 0],  # B01
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W_s = [
        [1, 0, 0, -1, 0, -1, 1], # C00
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [1, 0, 0, 0, -1, 1, -1]  # C11
    ]
    
    # Construct full matrices
    col_offset = 0
    for i_block in range(num_blocks):
        for k_block in range(num_blocks):
            for j_block in range(num_blocks):
                for c in range(7):
                    # U index: (i_block * 2 + i) * m + (k_block * 2 + j) ?
                    # A is n x m. A[i, j].
                    # Block i_block, k_block corresponds to A[i_block*2 + i, k_block*2 + j]?
                    # No, A is 6x6. Blocks are 2x2.
                    # A[i, j] where i = i_block*2 + i_local, j = k_block*2 + j_local?
                    # Wait, A is n x m. i in 0..5, j in 0..5.
                    # Block (i_block, k_block) is A[i_block*2 + i_local, k_block*2 + j_local].
                    # But Strassen multiplies A_block * B_block.
                    # A_block is 2x2. B_block is 2x2.
                    # A_block[i_local, j_local] corresponds to A[i_block*2 + i_local, k_block*2 + j_local].
                    # B_block[j_local, k_local] corresponds to B[k_block*2 + j_local, j_block*2 + k_local].
                    # C_block[i_local, k_local] corresponds to C[i_block*2 + i_local, j_block*2 + k_local].
                    
                    # U row: A index.
                    # V row: B index.
                    # W row: C index.
                    
                    for i_local in range(2):
                        for j_local in range(2):
                            u_val = U_s[i_local * 2 + j_local][c]
                            if u_val != 0:
                                row_u = (i_block * 2 + i_local) * m + (k_block * 2 + j_local)
                                U[row_u][col_offset] += u_val
                            
                    for j_local in range(2):
                        for k_local in range(2):
                            v_val = V_s[j_local * 2 + k_local][c]
                            if v_val != 0:
                                row_v = (k_block * 2 + j_local) * p + (j_block * 2 + k_local)
                                V[row_v][col_offset] += v_val
                            
                    for i_local in range(2):
                        for k_local in range(2):
                            w_val = W_s[k_local * 2 + i_local][c] # W index is k*n+i? No, W[k*n+i].
                            # C[i, j]. i = i_block*2 + i_local, j = j_block*2 + k_local.
                            # W index is j*n + i.
                            if w_val != 0:
                                row_w = (j_block * 2 + k_local) * n + (i_block * 2 + i_local)
                                W[row_w][col_offset] += w_val
                                
                    col_offset += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
