"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2_coefficients():
    """
    Returns U, V, W matrices for 2x2 matrix multiplication using Strassen.
    Shapes: (4, 7), (4, 7), (4, 7)
    Indices: 0:00, 1:01, 2:10, 3:11
    """
    # Strassen 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to fill W for all 7 columns based on the output formulas.
    # Let's define the output contributions for each M_k
    
    # M1 contributes to C11 (+1) and C22 (+1)
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 contributes to C21 (+1) and C22 (-1)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    W[3, 1] = -1
    
    # M3 contributes to C12 (+1) and C22 (+1)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    W[3, 2] = 1
    
    # M4 contributes to C11 (+1) and C21 (+1)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 contributes to C11 (-1) and C12 (+1)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1
    W[1, 4] = 1
    
    # M6 contributes to C22 (+1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 contributes to C11 (+1)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 4, 4, 8
    
    # Get 2x2 Strassen coefficients
    U2, V2, W2 = _strassen_2x2_coefficients()
    
    # We want to build 4x4 decomposition (rank 49)
    # Then duplicate for 4x8 (rank 98)
    
    # 4x4 indices:
    # A: 4x4 -> index a in 0..15. a = 2*row + col? No, standard row-major: a = row*4 + col.
    # But we are using block structure.
    # Block (br, bc) contains 2x2.
    # Global index for A[i,j]: i = 2*br + r, j = 2*bc + c.
    # Index in A_flat: i*4 + j.
    
    # We will build U4, V4, W4 of shape (16, 49)
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Outer Strassen loop (7 steps)
    for k_out in range(7):
        # U2[:, k_out] gives coeffs for X_k (2x2 matrix of blocks)
        # V2[:, k_out] gives coeffs for Y_k (2x2 matrix of blocks)
        # W2[:, k_out] gives coeffs for Z_k (2x2 matrix of blocks)
        
        # X_k is a 2x2 matrix where each entry is a linear combo of A blocks.
        # Let X_k[br, bc] = sum_{a_idx} U2[a_idx, k_out] * A_block[br, bc]
        # Wait, U2 is 4x7. The 4 indices correspond to the 4 blocks of the 2x2 matrix.
        # U2[0, k] -> block (0,0)
        # U2[1, k] -> block (0,1)
        # U2[2, k] -> block (1,0)
        # U2[3, k] -> block (1,1)
        
        # So X_k is formed by taking linear combinations of the 4 blocks of A.
        # Similarly Y_k from B.
        
        # Now we need to multiply X_k (2x2) by Y_k (2x2) to get Z_k (2x2).
        # We use inner Strassen for this.
        
        for k_in in range(7):
            col_idx = k_out * 7 + k_in
            
            # Inner Strassen step k_in
            # u_in = U2[:, k_in] (coeffs for X entries)
            # v_in = V2[:, k_in] (coeffs for Y entries)
            # w_in = W2[:, k_in] (coeffs for Z entries)
            
            # X_entry[br, bc] = sum_{block_idx} U2[block_idx, k_out] * A_block[br, bc]
            # Y_entry[br, bc] = sum_{block_idx} V2[block_idx, k_out] * B_block[br, bc]
            
            # The inner multiplication computes:
            # Z_entry[wr, wc] = w_in[wr*2+wc] * ( sum_{xr,xc} u_in[xr*2+xc] X_entry[xr,xc] ) * ( sum_{yr,yc} v_in[yr*2+yc] Y_entry[yr,yc] )
            
            # Substitute X and Y:
            # Term = w_val * ( sum_{xr,xc} u_val[xr,xc] sum_{block_idx} U2[block_idx, k_out] A_block[xr,xc] ) 
            #                  * ( sum_{yr,yc} v_val[yr,yc] sum_{block_idx} V2[block_idx, k_out] B_block[yr,yc] )
            
            # This is a product of two linear forms in A and B.
            # L_A = sum_{xr,xc, block_idx} u_in[xr*2+xc] * U2[block_idx, k_out] * A_block[xr,xc]
            # L_B = sum_{yr,yc, block_idx} v_in[yr*2+yc] * V2[block_idx, k_out] * B_block[yr,yc]
            
            # We need to populate U4, V4, W4 for this column col_idx.
            
            # Coefficients for A (U4)
            # For each global A index (i,j):
            # i = 2*br + r, j = 2*bc + c
            # A_block[br, bc] contains A[i,j] at local (r,c).
            # Wait, A_block[br, bc] is the 2x2 submatrix.
            # The entry A[i,j] is at local row r, local col c within block (br, bc).
            # In the linear form L_A, we sum over blocks and local positions.
            # L_A = sum_{br,bc} ( sum_{r,c} u_in[br*2+c? No, u_in is indexed by block position in the 2x2 matrix X_k] )
            
            # Let's clarify indices.
            # X_k is a 2x2 matrix of blocks.
            # X_k[br, bc] is a block (2x2 matrix).
            # u_in is a 4-vector corresponding to the 4 positions in the 2x2 matrix X_k.
            # u_in[0] -> X_k[0,0]
            # u_in[1] -> X_k[0,1]
            # u_in[2] -> X_k[1,0]
            # u_in[3] -> X_k[1,1]
            
            # So, L_A = sum_{br,bc} u_in[br*2+bc] * X_k[br, bc]
            # And X_k[br, bc] = sum_{block_idx} U2[block_idx, k_out] * A_block[block_idx]
            # Note: block_idx maps to (br, bc) in the 2x2 block structure.
            # block_idx 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1).
            
            # So, L_A = sum_{br,bc} u_in[br*2+bc] * ( sum_{block_idx} U2[block_idx, k_out] * A_block[block_idx] )
            # L_A = sum_{block_idx} ( sum_{br,bc} u_in[br*2+bc] * (1 if block_idx corresponds to (br,bc) else 0) * U2[block_idx, k_out] ) * A_block[block_idx]
            
            # Let coeff_A_block[block_idx] = sum_{br,bc} u_in[br*2+bc] * (1 if block_idx == br*2+bc else 0) * U2[block_idx, k_out]
            # Actually, u_in[br*2+bc] is non-zero only for specific br,bc.
            # For a fixed block_idx (say 0, which is br=0,bc=0), the term is u_in[0] * U2[0, k_out] * A_block[0].
            # So coeff_A_block[block_idx] = u_in[block_idx] * U2[block_idx, k_out].
            
            # Similarly for B:
            # coeff_B_block[block_idx] = v_in[block_idx] * V2[block_idx, k_out].
            
            # Now, L_A = sum_{block_idx} coeff_A_block[block_idx] * A_block[block_idx]
            # A_block[block_idx] is a 2x2 matrix.
            # A_block[0] contains A[0,0], A[0,1], A[1,0], A[1,1].
            # A_block[1] contains A[0,2], A[0,3], A[1,2], A[1,3].
            # A_block[2] contains A[2,0], A[2,1], A[3,0], A[3,1].
            # A_block[3] contains A[2,2], A[2,3], A[3,2], A[3,3].
            
            # We need to expand L_A into coefficients for each A[i,j].
            # For block_idx b:
            # If b=0: A[0,0], A[0,1], A[1,0], A[1,1] all get coeff coeff_A_block[0].
            # If b=1: A[0,2], A[0,3], A[1,2], A[1,3] all get coeff coeff_A_block[1].
            # etc.
            
            # Same for B.
            
            # For W:
            # Z_k is a 2x2 matrix of blocks.
            # Z_k[wr, wc] = w_in[wr*2+wc] * L_A * L_B
            # The output C is assembled from Z_k blocks.
            # C_block[wr, wc] += Z_k[wr, wc]
            
            # So for each output block (wr, wc), the coefficient is w_in[wr*2+wc].
            # This applies to all 4 entries in that block.
            
            # Let's populate the arrays
            
            # 1. Calculate coeff_A_block and coeff_B_block
            coeff_A_block = np.zeros(4)
            coeff_B_block = np.zeros(4)
            for b in range(4):
                coeff_A_block[b] = U2[b, k_in] * U2[b, k_out]
                coeff_B_block[b] = V2[b, k_in] * V2[b, k_out]
            
            # 2. Map to global indices for U4 and V4
            # U4 shape (16, 49). col_idx is the current column.
            
            for b in range(4):
                ca = coeff_A_block[b]
                cb = coeff_B_block[b]
                
                # Determine global indices for A_block[b]
                # b=0: rows 0,1 cols 0,1
                # b=1: rows 0,1 cols 2,3
                # b=2: rows 2,3 cols 0,1
                # b=3: rows 2,3 cols 2,3
                
                br = b // 2
                bc = b % 2
                
                for r in range(2):
                    for c in range(2):
                        i = 2*br + r
                        j = 2*bc + c
                        idx_a = i*4 + j
                        U4[idx_a, col_idx] += ca
                        
                        # For V4, the block structure is the same for B
                        # B is 4x4.
                        idx_b = i*4 + j
                        V4[idx_b, col_idx] += cb
                        
            # 3. Map to global indices for W4
            # W4 corresponds to C (4x4).
            # The output of this specific Strassen component (k_out, k_in) contributes to C.
            # The contribution is w_in[wr*2+wc] * L_A * L_B to C_block[wr, wc].
            
            for wr in range(2):
                for wc in range(2):
                    w_val = W2[wr*2+wc, k_in]
                    # Wait, W2 is from the inner Strassen.
                    # The inner Strassen computes Z_k = X_k Y_k.
                    # Z_k[wr, wc] = w_in[wr*2+wc] * (linear combo of X) * (linear combo of Y).
                    # So the coefficient for the block (wr, wc) is w_val.
                    
                    if w_val != 0:
                        for r in range(2):
                            for c in range(2):
                                i = 2*wr + r
                                j = 2*wc + c
                                idx_c = i*4 + j
                                W4[idx_c, col_idx] += w_val

    # Now we have U4, V4, W4 for 4x4 multiplication.
    # We need 4x8.
    # B is 4x8. Split into B_L (4x4) and B_R (4x4).
    # C is 4x8. Split into C_L (4x4) and C_R (4x4).
    
    # U is 4x4 (A). Shape (16, 98).
    # V is 4x8 (B). Shape (32, 98).
    # W is 4x8 (C). Shape (32, 98).
    
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    # First 49 columns: A * B_L -> C_L
    # A is same.
    U[:, 0:49] = U4
    
    # B_L is the first 16 entries of B (cols 0,1,2,3).
    # B is 4x8. Index b_idx = row*8 + col.
    # B_L[row, col] is at B[row*8 + col] for col in 0..3.
    # V4 is for 4x4 B. Index v_idx = row*4 + col.
    # We need to map V4 indices to V indices.
    
    for r in range(4):
        for c in range(4):
            v4_idx = r*4 + c
            v_idx = r*8 + c # B_L is left half
            V[v_idx, 0:49] = V4[v4_idx, 0:49]
            
    # C_L is the first 16 entries of C.
    for r in range(4):
        for c in range(4):
            w4_idx = r*4 + c
            w_idx = r*8 + c # C_L is left half
            W[w_idx, 0:49] = W4[w4_idx, 0:49]
            
    # Next 49 columns: A * B_R -> C_R
    U[:, 49:98] = U4
    
    # B_R is the second 16 entries of B (cols 4,5,6,7).
    for r in range(4):
        for c in range(4):
            v4_idx = r*4 + c
            v_idx = r*8 + (c + 4) # B_R is right half
            V[v_idx, 49:98] = V4[v4_idx, 0:49]
            
    # C_R is the second 16 entries of C.
    for r in range(4):
        for c in range(4):
            w4_idx = r*4 + c
            w_idx = r*8 + (c + 4) # C_R is right half
            W[w_idx, 49:98] = W4[w4_idx, 0:49]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
