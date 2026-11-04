import numpy as np

def get_strassen_2x2():
    # Returns U, V, W for 2x2x2 Strassen
    # U: 4x7, V: 4x7, W: 4x7
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A0+A3)(B0+B3)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A2+A3)B0
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A0(B1-B3)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A3(B2-B0)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A0+A1)B3
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A2-A0)(B0+B1)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A1-A3)(B2+B3)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # C0 = M1 - M4 + M5 - M7
    W[0, 0] = 1
    W[0, 3] = -1
    W[0, 4] = 1
    W[0, 6] = -1
    
    # C1 = M1 + M2
    W[1, 0] = 1
    W[1, 1] = 1
    
    # C2 = M3 + M4
    W[2, 2] = 1
    W[2, 3] = 1
    
    # C3 = M5 - M2 + M3 - M6
    W[3, 4] = 1
    W[3, 1] = -1
    W[3, 2] = 1
    W[3, 5] = -1
    
    return U, V, W

def construct():
    n, m, p = 4, 4, 5
    
    # Rank 49 part for 4x4x4
    U_49 = np.zeros((16, 49))
    V_49 = np.zeros((16, 49))
    W_49 = np.zeros((16, 49))
    
    U2, V2, W2 = get_strassen_2x2()
    
    # Define block indices for A and B (4x4)
    # A blocks:
    # A00: 0-3, A01: 4-7, A10: 8-11, A11: 12-15
    # B blocks:
    # B00: 0-3, B01: 4-7, B10: 8-11, B11: 12-15
    
    # Top-level Strassen steps
    # Each step k has:
    # - Coefficients for A blocks (4 blocks, each 2x2)
    # - Coefficients for B blocks (4 blocks, each 2x2)
    # - Coefficients for C blocks (4 blocks, each 2x2)
    
    # We represent the linear combination of blocks as a 4x4 matrix of coefficients?
    # No, each block is 2x2. The combination is element-wise.
    # So if M_k uses A00 with coeff 1, then all 4 elements of A00 are added with coeff 1.
    
    # Let's define the top-level coefficients for A and B blocks.
    # A_blocks: 0,1,2,3 corresponding to A00, A01, A10, A11
    # B_blocks: 0,1,2,3 corresponding to B00, B01, B10, B11
    
    top_steps = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M0 = (A00+A11)(B00+B11)
        ([1,0,0,1], [1,0,0,1], [1,0,0,0]), # C00
        # M1 = (A10+A11)B00
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]), # C10
        # M2 = A00(B01-B11)
        ([1,0,0,0], [0,1,0,-1], [0,0,1,0]), # C01
        # M3 = A11(B10-B00)
        ([0,0,0,1], [0,0,1,-1], [0,0,0,1]), # C11
        # M4 = (A00+A01)B11
        ([1,1,0,0], [0,0,0,1], [0,0,0,1]), # C11
        # M5 = (A10-A00)(B00+B01)
        ([-1,0,1,0], [1,1,0,0], [0,1,0,0]), # C10
        # M6 = (A01-A11)(B10+B11)
        ([0,1,0,-1], [0,0,1,1], [0,0,1,0]), # C01
    ]
    
    # Wait, the C coefficients need to be mapped correctly.
    # Top-level C blocks:
    # C00 = M0 - M3 + M4 - M6
    # C01 = M2 + M3
    # C10 = M0 + M1
    # C11 = M4 - M1 + M2 - M5
    
    # Let's re-verify the top-level Strassen formulas for blocks.
    # C00 (block 0) = M0 - M3 + M4 - M6
    # C01 (block 1) = M2 + M3
    # C10 (block 2) = M0 + M1
    # C11 (block 3) = M4 - M1 + M2 - M5
    
    # So for each M_k, we need to add its contribution to the appropriate C blocks.
    # M0 contributes to C00 (coeff 1) and C10 (coeff 1).
    # M1 contributes to C10 (coeff 1) and C11 (coeff -1).
    # M2 contributes to C01 (coeff 1) and C11 (coeff 1).
    # M3 contributes to C00 (coeff -1), C01 (coeff 1), C11 (coeff 1).
    # M4 contributes to C00 (coeff 1), C11 (coeff 1).
    # M5 contributes to C11 (coeff -1).
    # M6 contributes to C00 (coeff -1).
    
    # Let's store the C coefficients for each M_k.
    # C_blocks: 0,1,2,3
    c_coeffs = [
        [1, 0, 1, 0], # M0 -> C00, C10
        [0, 0, 1, -1], # M1 -> C10, C11
        [0, 1, 0, 1], # M2 -> C01, C11
        [-1, 1, 0, 1], # M3 -> C00, C01, C11
        [1, 0, 0, 1], # M4 -> C00, C11
        [0, 0, 0, -1], # M5 -> C11
        [-1, 0, 0, 0], # M6 -> C00
    ]
    
    # A_coeffs for each M_k (4 blocks)
    a_coeffs = [
        [1, 0, 0, 1], # M0
        [0, 0, 1, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 0, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0], # M5
        [0, 1, 0, -1], # M6
    ]
    
    # B_coeffs for each M_k (4 blocks)
    b_coeffs = [
        [1, 0, 0, 1], # M0
        [1, 0, 0, 0], # M1
        [0, 1, 0, -1], # M2
        [0, 0, 1, -1], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, 1], # M6
    ]
    
    col = 0
    for k in range(7):
        # Build L_A and L_B (4x16 matrices)
        L_A = np.zeros((4, 16))
        L_B = np.zeros((4, 16))
        
        for b in range(4):
            coeff = a_coeffs[k][b]
            if coeff != 0:
                # Block b corresponds to indices 4*b to 4*b+3
                L_A[:, 4*b:4*b+4] = coeff
        
            coeff = b_coeffs[k][b]
            if coeff != 0:
                L_B[:, 4*b:4*b+4] = coeff
        
        # Expand M_k using Strassen 2x2
        # U_k = U2 @ L_A (4x7 * 4x16 -> 4x16? No)
        # U2 is 4x7. L_A is 4x16.
        # The local A is 4x1. The global A is 16x1.
        # Local A_i = sum_j L_A[i,j] Global A_j.
        # The scalar mult for U2 is sum_i U2[i,m] Local A_i.
        # = sum_i U2[i,m] sum_j L_A[i,j] Global A_j
        # = sum_j (sum_i U2[i,m] L_A[i,j]) Global A_j
        # So U_global = U2.T @ L_A ?
        # U2 is 4x7. L_A is 4x16.
        # We want a 16x7 matrix.
        # U_global[j, m] = sum_i L_A[i, j] U2[i, m]
        # This is L_A.T @ U2.
        
        U_k = L_A.T @ U2 # 16x7
        V_k = L_B.T @ V2 # 16x7
        
        # W part
        # Local C is 4x1.
        # Local C_i = sum_m W2[i, m] M_k_m
        # Global C block b gets contribution c_coeffs[k][b] * Local C
        # Global C index for block b is 4*b to 4*b+3.
        # W_global[4*b + i, m] += c_coeffs[k][b] * W2[i, m]
        
        for b in range(4):
            coeff = c_coeffs[k][b]
            if coeff != 0:
                W_49[4*b:4*b+4, col:col+7] += coeff * W2
        
        U_49[:, col:col+7] = U_k
        V_49[:, col:col+7] = V_k
        col += 7
    
    # Rank 16 part for 4x4x1
    # Schoolbook for A (4x4) * B_col (4x1)
    # C (4x1)
    # C_i = sum_j A_ij B_j
    # Rank 16.
    U_16 = np.zeros((16, 16))
    V_16 = np.zeros((16, 16)) # B is 4x1, but V shape is m*p. Here p=1 for this part?
    # The full V is 4x5. The last column is index 16..19?
    # No, V is m*p = 4*5 = 20.
    # The 4x4x1 part uses the last column of B (index 16..19 in V? No, B is 4x5.
    # B indices: 0..19.
    # B_col (4x1) is the 5th column. Indices 16,17,18,19.
    # Wait, B is 4x5.
    # B[j, k]. j=0..3, k=0..4.
    # Index in V: j*5 + k.
    # Last column k=4. Indices 4, 9, 14, 19.
    
    # So V_16 should have non-zeros at indices 4, 9, 14, 19.
    
    # Let's construct U_16, V_16, W_16 for the 4x4x1 part.
    # U: 16x16.
    # V: 20x16.
    # W: 20x16. (C is 4x5. We are computing the last column C[:, 4].
    # C indices: k*n + i. k=4. Indices 16, 17, 18, 19.
    
    U_16 = np.zeros((16, 16))
    V_16 = np.zeros((20, 16))
    W_16 = np.zeros((20, 16))
    
    col2 = 0
    for i in range(4):
        for j in range(4):
            U_16[i*4+j, col2] = 1
            V_16[j*5+4, col2] = 1 # B[j, 4]
            W_16[4*4+i, col2] = 1 # C[i, 4] -> index 16+i
            col2 += 1
    
    # Combine
    U = np.hstack([U_49, U_16])
    V = np.hstack([V_49, V_16])
    W = np.hstack([W_49, W_16])
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
