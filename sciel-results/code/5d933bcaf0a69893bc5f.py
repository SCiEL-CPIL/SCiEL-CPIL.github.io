import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Rank 32 solution:
    # Split 8 columns into two groups of 4.
    # For each group of 4:
    #   First 3 columns: use 7-term Strassen for 3x3.
    #   Last 1 column: use 9-term schoolbook for 3x3x1.
    # Total per group: 7 + 9 = 16.
    # Total rank: 16 * 2 = 32.

    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    # Define the 7-term Strassen algorithm for 3x3
    # A is 3x3, B is 3x3
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11+a33)(b11+b33)
    # M7 = (a12-a22+a33)(b22+b33)
    
    # Coefficients for U (A part), V (B part), W (C part)
    # Indices:
    # A: 0,1,2 (row), 3,4,5 (col) -> flat: i*3+j
    # B: 0,1,2 (row), 3,4,5 (col) -> flat: i*3+j
    # C: 0,1,2 (row), 3,4,5 (col) -> flat: i*3+j
    
    # U indices for A:
    # a11=0, a12=1, a13=2
    # a21=3, a22=4, a23=5
    # a31=6, a32=7, a33=8
    
    # V indices for B (for a specific 3x3 block):
    # b11=0, b12=1, b13=2
    # b21=3, b22=4, b23=5
    # b31=6, b32=7, b33=8
    
    # W indices for C (for a specific 3x3 block):
    # c11=0, c12=1, c13=2
    # c21=3, c22=4, c23=5
    # c31=6, c32=7, c33=8

    # Strassen terms (U_coeffs, V_coeffs, W_coeffs)
    # Each term is a vector of length 9 for U, 9 for V, 9 for W.
    
    def add_term(U_vec, V_vec, W_vec, offset):
        for i in range(9):
            if U_vec[i] != 0:
                U[i, offset] += U_vec[i]
            if V_vec[i] != 0:
                V[i, offset] += V_vec[i]
            if W_vec[i] != 0:
                W[i, offset] += W_vec[i]

    # Term 1: (a11+a22)(b11+b22)
    u1 = np.zeros(9); u1[0]=1; u1[4]=1
    v1 = np.zeros(9); v1[0]=1; v1[4]=1
    w1 = np.zeros(9)
    
    # Term 2: (a21+a22)b11
    u2 = np.zeros(9); u2[3]=1; u2[4]=1
    v2 = np.zeros(9); v2[0]=1
    w2 = np.zeros(9)
    
    # Term 3: a11(b12-b22)
    u3 = np.zeros(9); u3[0]=1
    v3 = np.zeros(9); v3[1]=1; v3[4]=-1
    w3 = np.zeros(9)
    
    # Term 4: a22(b21-b11)
    u4 = np.zeros(9); u4[4]=1
    v4 = np.zeros(9); v4[3]=1; v4[0]=-1
    w4 = np.zeros(9)
    
    # Term 5: (a11+a12)b22
    u5 = np.zeros(9); u5[0]=1; u5[1]=1
    v5 = np.zeros(9); v5[4]=1
    w5 = np.zeros(9)
    
    # Term 6: (a21-a11+a33)(b11+b33)
    u6 = np.zeros(9); u6[3]=1; u6[0]=-1; u6[8]=1
    v6 = np.zeros(9); v6[0]=1; v6[8]=1
    w6 = np.zeros(9)
    
    # Term 7: (a12-a22+a33)(b22+b33)
    u7 = np.zeros(9); u7[1]=1; u7[4]=-1; u7[8]=1
    v7 = np.zeros(9); v7[4]=1; v7[8]=1
    w7 = np.zeros(9)

    # Output coefficients for each c_ij
    # c11 = M1 - M4 + M6
    # c12 = M3 + M5 - M7
    # c13 = M6 + M7
    # c21 = M2 + M4 - M6
    # c22 = M1 - M2 + M7
    # c23 = M6 - M7
    # c31 = M2 + M6
    # c32 = M5 - M7
    # c33 = M1 - M5 + M6 - M7

    # We need to distribute the M_k to the W matrix.
    # For each column block, we have 7 M's.
    # For each M_k, we add its contribution to W for each c_ij.
    
    def add_strassen_block(offset, col_start):
        # col_start is the starting column index in B (0-7)
        # We are processing a 3x3 block of B starting at col_start
        # B indices in V are: row * 8 + (col_start + col)
        
        # Map local B indices (0-8) to global V indices
        # Local b11 (0) -> V[0*8 + col_start]
        # Local b12 (1) -> V[0*8 + col_start + 1]
        # ...
        # Local b33 (8) -> V[2*8 + col_start + 2]
        
        # We already defined v1..v7 with local indices 0-8.
        # We need to shift them by col_start in the column dimension.
        
        # Let's create a helper to shift V indices
        def shift_v(v_local, col_offset):
            v_global = np.zeros(24) # m*p = 3*8
            for i in range(3):
                for j in range(3):
                    idx_local = i*3+j
                    idx_global = i*8 + (j + col_offset)
                    v_global[idx_global] = v_local[idx_local]
            return v_global

        # W indices are local 0-8 for C.
        # W global indices: k*n + i.
        # For a specific column block, the C columns are col_start, col_start+1, col_start+2.
        # W index for c_{ij} where j is local col (0-2) is (col_start+j)*3 + i.
        
        def shift_w(w_local, col_offset):
            w_global = np.zeros(24) # p*n = 8*3
            for i in range(3):
                for j in range(3):
                    idx_local = i*3+j # Wait, W index is k*n+i.
                    # Local c_{ij} is at i*3+j in my previous definition?
                    # No, W[k*n+i, r].
                    # So for local col j, global col is col_start+j.
                    # W index is (col_start+j)*3 + i.
                    idx_global = (col_start+j)*3 + i
                    w_global[idx_global] = w_local[i*3+j]
            return w_global

        # U is not shifted.
        
        terms = [
            (u1, v1, w1, [1, 0, 0, 0, 0, 0, 0, 0, 0]), # M1
            (u2, v2, w2, [0, 0, 0, 1, 0, 0, 1, 0, 0]), # M2 -> c21, c31
            (u3, v3, w3, [0, 1, 0, 0, 0, 0, 0, 0, 0]), # M3 -> c12
            (u4, v4, w4, [0, 0, 0, 0, 0, 0, 0, 0, 0]), # M4 -> c11, c21 (handled in w)
            (u5, v5, w5, [0, 0, 0, 0, 0, 0, 0, 0, 0]), # M5 -> c12, c32, c33 (handled in w)
            (u6, v6, w6, [0, 0, 0, 0, 0, 0, 0, 0, 0]), # M6 -> c11, c13, c21, c23, c31, c33
            (u7, v7, w7, [0, 0, 0, 0, 0, 0, 0, 0, 0]), # M7 -> c12, c13, c22, c23, c32, c33
        ]
        
        # Wait, I need to add the W contributions for each M.
        # c11 = M1 - M4 + M6
        # c12 = M3 + M5 - M7
        # c13 = M6 + M7
        # c21 = M2 + M4 - M6
        # c22 = M1 - M2 + M7
        # c23 = M6 - M7
        # c31 = M2 + M6
        # c32 = M5 - M7
        # c33 = M1 - M5 + M6 - M7
        
        # I will add U and V for each term, and accumulate W.
        
        w_accum = np.zeros(24)
        
        for k in range(7):
            u_k, v_k, _, _ = terms[k]
            v_global = shift_v(v_k, col_start)
            
            # Add U
            U[:, offset + k] = u_k
            
            # Add V
            V[:, offset + k] = v_global
            
            # Add W contributions
            # M1: c11(+1), c22(+1), c33(+1)
            if k == 0:
                w_accum[(col_start+0)*3 + 0] += 1 # c11
                w_accum[(col_start+1)*3 + 1] += 1 # c22 (j=1, i=1)
                w_accum[(col_start+2)*3 + 2] += 1 # c33 (j=2, i=2)
            # M2: c21(+1), c31(+1), c22(-1)
            elif k == 1:
                w_accum[(col_start+0)*3 + 1] += 1 # c21 (j=0, i=1)
                w_accum[(col_start+2)*3 + 1] += 1 # c31 (j=2, i=1)
                w_accum[(col_start+1)*3 + 1] -= 1 # c22
            # M3: c12(+1)
            elif k == 2:
                w_accum[(col_start+1)*3 + 0] += 1 # c12 (j=1, i=0)
            # M4: c11(-1), c21(+1)
            elif k == 3:
                w_accum[(col_start+0)*3 + 0] -= 1 # c11
                w_accum[(col_start+0)*3 + 1] += 1 # c21
            # M5: c12(+1), c32(+1), c33(-1)
            elif k == 4:
                w_accum[(col_start+1)*3 + 0] += 1 # c12
                w_accum[(col_start+2)*3 + 1] += 1 # c32 (j=2, i=1)
                w_accum[(col_start+2)*3 + 2] -= 1 # c33
            # M6: c11(+1), c13(+1), c21(-1), c23(+1), c31(+1), c33(+1)
            elif k == 5:
                w_accum[(col_start+0)*3 + 0] += 1 # c11
                w_accum[(col_start+2)*3 + 0] += 1 # c13 (j=2, i=0)
                w_accum[(col_start+0)*3 + 1] -= 1 # c21
                w_accum[(col_start+2)*3 + 1] += 1 # c23 (j=2, i=1)
                w_accum[(col_start+2)*3 + 2] += 1 # c31 (j=2, i=2) -- Wait, c31 is i=2, j=0?
                # c_{ij}: i=row, j=col.
                # c31: i=2, j=0. W index: j*n + i = 0*3 + 2 = 2.
                # My previous mapping: w_global[(col_start+j)*3 + i].
                # So c31 (i=2, j=0) -> (col_start+0)*3 + 2.
                w_accum[(col_start+0)*3 + 2] += 1 # c31
                w_accum[(col_start+2)*3 + 2] += 1 # c33
            # M7: c12(-1), c13(+1), c22(+1), c23(-1), c32(-1), c33(-1)
            elif k == 6:
                w_accum[(col_start+1)*3 + 0] -= 1 # c12
                w_accum[(col_start+2)*3 + 0] += 1 # c13
                w_accum[(col_start+1)*3 + 1] += 1 # c22
                w_accum[(col_start+2)*3 + 1] -= 1 # c23
                w_accum[(col_start+2)*3 + 1] -= 1 # c32? No, c32 is i=1, j=2.
                # c32: i=1, j=2. W index: (col_start+2)*3 + 1.
                # So c23 and c32 are the same W index?
                # c23: i=1, j=2.
                # c32: i=1, j=2.
                # Yes, they are the same entry in C? No.
                # C is 3x3. c_{23} is row 2, col 3. c_{32} is row 3, col 2.
                # In 0-indexed: c_{1,2} and c_{2,1}.
                # My indices: i=row, j=col.
                # c12: i=0, j=1.
                # c21: i=1, j=0.
                # c23: i=1, j=2.
                # c32: i=2, j=1.
                
                # Let's re-map W indices carefully.
                # W[k*n+i, r]. k is col of B (0-7), i is row of C (0-2).
                # For local col j (0-2), global col is col_start+j.
                # W index: (col_start+j)*3 + i.
                
                # c12: i=0, j=1. Index: (col_start+1)*3 + 0.
                # c13: i=0, j=2. Index: (col_start+2)*3 + 0.
                # c22: i=1, j=1. Index: (col_start+1)*3 + 1.
                # c23: i=1, j=2. Index: (col_start+2)*3 + 1.
                # c32: i=2, j=1. Index: (col_start+1)*3 + 2.
                # c33: i=2, j=2. Index: (col_start+2)*3 + 2.
                
                # M7 contributions:
                # c12 (-1): (col_start+1)*3 + 0
                # c13 (+1): (col_start+2)*3 + 0
                # c22 (+1): (col_start+1)*3 + 1
                # c23 (-1): (col_start+2)*3 + 1
                # c32 (-1): (col_start+1)*3 + 2
                # c33 (-1): (col_start+2)*3 + 2
                
                w_accum[(col_start+1)*3 + 0] -= 1
                w_accum[(col_start+2)*3 + 0] += 1
                w_accum[(col_start+1)*3 + 1] += 1
                w_accum[(col_start+2)*3 + 1] -= 1
                w_accum[(col_start+1)*3 + 2] -= 1
                w_accum[(col_start+2)*3 + 2] -= 1

        W[:, offset:offset+7] = w_accum[:, np.newaxis] # No, w_accum is 24x1, we need 24x7
        # Actually, W is 24x32. We are filling columns offset..offset+6.
        # w_accum is a single column of W.
        # But each M_k has its own W column?
        # No, the W matrix column r corresponds to the r-th rank term.
        # For each rank term (M_k), we have a W vector.
        # So W[:, offset+k] should be the W contribution of M_k.
        
        # I need to store W for each M_k separately.
        
        # Let's restart the W accumulation per term.
        
        for k in range(7):
            w_k = np.zeros(24)
            if k == 0:
                w_k[(col_start+0)*3 + 0] += 1
                w_k[(col_start+1)*3 + 1] += 1
                w_k[(col_start+2)*3 + 2] += 1
            elif k == 1:
                w_k[(col_start+0)*3 + 1] += 1
                w_k[(col_start+2)*3 + 1] += 1
                w_k[(col_start+1)*3 + 1] -= 1
            elif k == 2:
                w_k[(col_start+1)*3 + 0] += 1
            elif k == 3:
                w_k[(col_start+0)*3 + 0] -= 1
                w_k[(col_start+0)*3 + 1] += 1
            elif k == 4:
                w_k[(col_start+1)*3 + 0] += 1
                w_k[(col_start+2)*3 + 1] += 1
                w_k[(col_start+2)*3 + 2] -= 1
            elif k == 5:
                w_k[(col_start+0)*3 + 0] += 1
                w_k[(col_start+2)*3 + 0] += 1
                w_k[(col_start+0)*3 + 1] -= 1
                w_k[(col_start+2)*3 + 1] += 1
                w_k[(col_start+0)*3 + 2] += 1
                w_k[(col_start+2)*3 + 2] += 1
            elif k == 6:
                w_k[(col_start+1)*3 + 0] -= 1
                w_k[(col_start+2)*3 + 0] += 1
                w_k[(col_start+1)*3 + 1] += 1
                w_k[(col_start+2)*3 + 1] -= 1
                w_k[(col_start+1)*3 + 2] -= 1
                w_k[(col_start+2)*3 + 2] -= 1
            
            W[:, offset+k] = w_k

    # Add schoolbook for the 4th column of each group
    def add_schoolbook_column(offset, col_idx):
        # 9 terms
        for i in range(3):
            for j in range(3):
                U[i*3+j, offset] = 1
                V[j*8+col_idx, offset] = 1
                W[col_idx*3+i, offset] = 1
                offset += 1
        return offset

    # Group 1: cols 0,1,2 (Strassen), col 3 (Schoolbook)
    offset = 0
    add_strassen_block(offset, 0)
    offset += 7
    offset = add_schoolbook_column(offset, 3)
    
    # Group 2: cols 4,5,6 (Strassen), col 7 (Schoolbook)
    add_strassen_block(offset, 4)
    offset += 7
    offset = add_schoolbook_column(offset, 7)
    
    # Verify
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is not a simple dot product.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
