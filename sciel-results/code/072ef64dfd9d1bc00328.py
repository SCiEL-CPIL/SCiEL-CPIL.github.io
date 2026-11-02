import numpy as np

def _strassen_2x2_decomp():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indices:
    U[i*2+j, c]
    V[j*2+k, c]
    W[k*2+i, c]
    """
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # M1 = (A11+A22)(B11+B22)
    # A parts: A11, A22
    # B parts: B11, B22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # Result M1 contributes to C11, C22
    # C11 += M1 => W[11] += 1. Index 11 in W is k=1, i=1? 
    # W index: k*2+i. 
    # C11 is i=1, k=1? No, C[i,k]. 
    # C11 is i=0, k=0. W[0*2+0] = W[0].
    # C22 is i=1, k=1. W[1*2+1] = W[3].
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A21+A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2 => i=1, k=0. W[0*2+1] = W[1].
    W[1, 1] = 1
    # C22 -= M2 => i=1, k=1. W[3] -= 1.
    W[3, 1] = -1
    
    # M3 = A11(B12-B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3 => i=0, k=1. W[1*2+0] = W[2].
    W[2, 2] = 1
    # C22 += M3 => i=1, k=1. W[3] += 1.
    W[3, 2] = 1
    
    # M4 = A22(B21-B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4 => i=0, k=0. W[0] += 1.
    W[0, 3] = 1
    # C21 += M4 => i=1, k=0. W[1] += 1.
    W[1, 3] = 1
    
    # M5 = (A11+A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5 => i=0, k=0. W[0] -= 1.
    W[0, 4] = -1
    # C12 += M5 => i=0, k=1. W[2] += 1.
    W[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6 => i=1, k=1. W[3] += 1.
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7 => i=0, k=0. W[0] += 1.
    W[0, 6] = 1
    
    return U, V, W

def _strassen_4x4_decomp():
    """
    Returns U, V, W for 4x4x4 Strassen decomposition.
    U shape (16, 49), V shape (16, 49), W shape (16, 49).
    """
    r = 49
    U = np.zeros((16, r))
    V = np.zeros((16, r))
    W = np.zeros((16, r))
    
    U2, V2, W2 = _strassen_2x2_decomp()
    
    # The 4x4x4 algorithm uses 7 products of 2x2 matrices.
    # Let's define the linear combinations of A and B blocks for the 7 macro-terms.
    # A blocks: A00, A01, A10, A11 (indices 0..3 in 2x2 block grid)
    # B blocks: B00, B01, B10, B11
    
    # Macro term 1: (A00+A11)(B00+B11)
    # Macro term 2: (A10+A11)B00
    # Macro term 3: A00(B01-B11)
    # Macro term 4: A11(B10-B00)
    # Macro term 5: (A00+A01)B11
    # Macro term 6: (A10-A00)(B00+B01)
    # Macro term 7: (A01-A11)(B10+B11)
    
    # For each macro term t (0..6), we expand it into 7 scalar multiplications.
    # The expansion uses U2, V2, W2.
    # The "A" part of the macro term is a linear combination of A blocks.
    # The "B" part of the macro term is a linear combination of B blocks.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
    # In the flat index i*4+j, the indices are:
    # Block (0,0): 0,1,4,5
    # Block (0,1): 2,3,6,7
    # Block (1,0): 8,9,12,13
    # Block (1,1): 10,11,14,15
    
    # Let's define the coefficients for A and B for each macro term.
    # A_coeffs[t][block_idx]
    # B_coeffs[t][block_idx]
    
    A_coeffs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1},       # M3
        {3: 1},       # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1},# M6
        {1: 1, 3: -1} # M7
    ]
    
    B_coeffs = [
        {0: 1, 3: 1}, # M1
        {0: 1},       # M2
        {1: 1, 3: -1},# M3
        {2: 1, 0: -1},# M4
        {3: 1},       # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1}  # M7
    ]
    
    # Output coefficients for the 4 output blocks C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_coeffs = [
        {0: 1, 3: 1, 4: -1, 6: 1}, # C00
        {2: 1, 4: 1},               # C01
        {1: 1, 3: 1},               # C10
        {0: 1, 1: -1, 2: 1, 5: 1}  # C11
    ]
    
    # Helper to get flat indices for a 2x2 block
    def block_indices(bi, bj):
        return [2*bi*4 + 2*bj, 2*bi*4 + 2*bj + 1, 2*bi*4 + 2*bj + 4, 2*bi*4 + 2*bj + 5]
    
    # We iterate over the 7 macro terms
    for t in range(7):
        # For each of the 7 scalar multiplications in the 2x2x2 decomp
        for c2 in range(7):
            c = t * 7 + c2
            
            # A part
            # U2[i2*2+j2, c2] is the coefficient for A[i2, j2] in the c2-th product of the 2x2x2 decomp.
            # We need to map this to the 4x4 indices.
            # The "A" in the macro term is a linear combination of A blocks.
            # So the coefficient for A_flat[idx] is sum over blocks b of A_coeffs[t][b] * U2[idx_in_block, c2].
            
            for i2 in range(2):
                for j2 in range(2):
                    u_val = U2[i2*2+j2, c2]
                    if u_val == 0: continue
                    # This u_val applies to all A blocks b where A_coeffs[t][b] != 0
                    # Wait, U2 is for the 2x2 matrix "A" of the macro term.
                    # The macro term "A" is formed by combining A blocks.
                    # So if A_coeffs[t][b] is non-zero, then the element A[i2, j2] of block b contributes.
                    for b in A_coeffs[t]:
                        coeff = A_coeffs[t][b]
                        bi, bj = b // 2, b % 2
                        idxs = block_indices(bi, bj)
                        # Map i2, j2 to local index in block
                        # i2 is 0 or 1, j2 is 0 or 1
                        local_idx = i2*2 + j2
                        flat_idx = idxs[local_idx]
                        U[flat_idx, c] += u_val * coeff
            
            # B part
            for j2 in range(2):
                for k2 in range(2):
                    v_val = V2[j2*2+k2, c2]
                    if v_val == 0: continue
                    for b in B_coeffs[t]:
                        coeff = B_coeffs[t][b]
                        bi, bj = b // 2, b % 2
                        idxs = block_indices(bi, bj)
                        local_idx = j2*2 + k2
                        flat_idx = idxs[local_idx]
                        V[flat_idx, c] += v_val * coeff
            
            # W part
            # The result of the macro term is a 2x2 matrix.
            # It contributes to the output blocks C according to C_coeffs.
            # W2[k2*2+i2, c2] is the coefficient for C[i2, k2] of the macro term result.
            for k2 in range(2):
                for i2 in range(2):
                    w_val = W2[k2*2+i2, c2]
                    if w_val == 0: continue
                    # This w_val applies to all output blocks c_out where C_coeffs[c_out][t] != 0
                    for c_out in C_coeffs:
                        if t in C_coeffs[c_out]:
                            coeff = C_coeffs[c_out][t]
                            ci, cj = c_out // 2, c_out % 2
                            idxs = block_indices(ci, cj)
                            local_idx = i2*2 + k2
                            flat_idx = idxs[local_idx]
                            W[flat_idx, c] += w_val * coeff

    return U, V, W

def construct():
    n, m, p = 4, 4, 8
    # Get 4x4x4 decomposition
    U4, V4, W4 = _strassen_4x4_decomp()
    r4 = U4.shape[1] # 49
    
    r = 2 * r4 # 98
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # First half: B_L (k=0..3)
    U[:, 0:r4] = U4
    
    # V mapping:
    # V4 indices: j*4 + k (j=0..3, k=0..3)
    # V indices: j*8 + k (j=0..3, k=0..7)
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:r4] = V4[j*4 + k, :]
    
    # W mapping:
    # W4 indices: k*4 + i (k=0..3, i=0..3)
    # W indices: k*4 + i (k=0..7, i=0..3)
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:r4] = W4[k*4 + i, :]
            
    # Second half: B_R (k=4..7)
    U[:, r4:2*r4] = U4
    
    # V mapping:
    # V4 indices: j*4 + k_local (k_local=0..3)
    # V indices: j*8 + k_global (k_global=4..7)
    for j in range(4):
        for k_local in range(4):
            k_global = k_local + 4
            V[j*8 + k_global, r4:2*r4] = V4[j*4 + k_local, :]
            
    # W mapping:
    # W4 indices: k_local*4 + i
    # W indices: k_global*4 + i
    for k_local in range(4):
        k_global = k_local + 4
        for i in range(4):
            W[k_global*4 + i, r4:2*r4] = W4[k_local*4 + i, :]
            
    return {"U": U, "V": V, "W": W}
