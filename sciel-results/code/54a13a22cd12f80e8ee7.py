import numpy as np

def construct():
    n, m, p = 4, 4, 7
    
    # 1. M_4,4,4 part (rank 49)
    # Use Kronecker product of Strassen 2x2x2
    # Strassen 2x2x2 matrices U, V, W of shape 4x7
    # We need to define them correctly for the tensor T[i*2+j, j*2+k, k*2+i]
    
    # Let's build the Strassen decomposition explicitly.
    # Indices for 2x2: i,j,k in {0,1}
    # a = 2*i + j
    # b = 2*j + k
    # c = 2*k + i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00 + A11)(B00 + B11) -> C00 +=, C11 +=, C01? No.
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M6
    # C11 = M1 - M2 + M4 + M5
    
    # We need to fill U2, V2, W2 such that sum U[:,r] V[:,r] W[:,r] = T.
    # T[a,b,c] = 1 if a=2i+j, b=2j+k, c=2k+i.
    
    # For each term r, we add +/- 1 to U2, V2, W2 at specific indices.
    
    # Helper to add a term
    def add_term(r, u_idx, v_idx, w_idx, sign=1):
        U2[u_idx, r] += sign
        V2[v_idx, r] += sign
        W2[w_idx, r] += sign

    # Term 1: M1 = (A00 + A11)(B00 + B11)
    # A00 is a=0, B00 is b=0, C?
    # M1 contributes to C00 (c=0) and C11 (c=3).
    # C00: i=0, k=0 -> c=0.
    # C11: i=1, k=1 -> c=3.
    
    # Wait, the tensor decomposition is a sum of rank 1 tensors.
    # Each rank 1 tensor is u v^T w^T.
    # The entries are products.
    # We just need to define the vectors u, v, w for each of the 7 terms.
    
    # Term 1:
    # u: A00 (a=0), A11 (a=3)
    # v: B00 (b=0), B11 (b=3)
    # w: C00 (c=0), C11 (c=3)
    # But it's not a single rank 1 term for the whole product.
    # Strassen's algorithm expresses the multiplication as 7 multiplications of linear combinations.
    # The tensor decomposition corresponds to these 7 multiplications.
    
    # Let's verify the indices for Term 1:
    # u1: 1 at a=0 (A00), 1 at a=3 (A11).
    # v1: 1 at b=0 (B00), 1 at b=3 (B11).
    # w1: 1 at c=0 (C00), 1 at c=3 (C11).
    # This term represents (A00+A11)(B00+B11).
    # In the expansion C00 = ... - M4 + M5 - M7 + M1, M1 has coeff 1.
    # In C11 = ... - M2 + M4 + M5 + M1, M1 has coeff 1.
    # So w1 should have 1 at c=0 and 1 at c=3.
    
    U2[0, 0] = 1
    U2[3, 0] = 1
    V2[0, 0] = 1
    V2[3, 0] = 1
    W2[0, 0] = 1
    W2[3, 0] = 1
    
    # Term 2: M2 = (A10 + A11)B00
    # A10 is a=2, A11 is a=3.
    # B00 is b=0.
    # Contributes to C10 (c=2) with coeff 1.
    U2[2, 1] = 1
    U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1
    
    # Term 3: M3 = A00(B10 - B11)
    # A00 is a=0.
    # B10 is b=2, B11 is b=3.
    # Contributes to C01 (c=1) with coeff 1.
    U2[0, 2] = 1
    V2[2, 2] = 1
    V2[3, 2] = -1
    W2[1, 2] = 1
    
    # Term 4: M4 = A11(B01 - B00)
    # A11 is a=3.
    # B01 is b=1, B00 is b=0.
    # Contributes to C00 (c=0) with coeff -1, and C11 (c=3) with coeff 1.
    U2[3, 3] = 1
    V2[1, 3] = 1
    V2[0, 3] = -1
    W2[0, 3] = -1
    W2[3, 3] = 1
    
    # Term 5: M5 = (A00 + A01)B11
    # A00 is a=0, A01 is a=1.
    # B11 is b=3.
    # Contributes to C00 (c=0) with coeff 1, and C11 (c=3) with coeff 1.
    U2[0, 4] = 1
    U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1
    W2[3, 4] = 1
    
    # Term 6: M6 = (A10 - A00)(B00 + B01)
    # A10 is a=2, A00 is a=0.
    # B00 is b=0, B01 is b=1.
    # Contributes to C10 (c=2) with coeff 1.
    U2[2, 5] = 1
    U2[0, 5] = -1
    V2[0, 5] = 1
    V2[1, 5] = 1
    W2[2, 5] = 1
    
    # Term 7: M7 = (A01 - A11)(B10 + B11)
    # A01 is a=1, A11 is a=3.
    # B10 is b=2, B11 is b=3.
    # Contributes to C01 (c=1) with coeff 1, and C00 (c=0) with coeff -1.
    U2[1, 6] = 1
    U2[3, 6] = -1
    V2[2, 6] = 1
    V2[3, 6] = 1
    W2[1, 6] = 1
    W2[0, 6] = -1
    
    # Now U2, V2, W2 are 4x7.
    # Check if they form a valid decomposition for M_2,2,2.
    # T2 = sum U2[:,r] outer V2[:,r] outer W2[:,r]
    
    # Now for M_4,4,4, we use Kronecker products.
    # U4 = U2 kron U2
    # V4 = V2 kron V2
    # W4 = W2 kron W2
    # This gives rank 49.
    
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # 2. M_4,4,2 part (rank 28)
    # M_4,4,2 = M_2,4,2 (top half of A) + M_2,4,2 (bottom half of A)
    # M_2,4,2 rank 14 (two Strassen blocks for B's columns)
    # B is 4x2. Split B into B_col0 (4x1) and B_col1 (4x1)? No.
    # B is 4x2.
    # M_2,4,2: A is 2x4, B is 4x2.
    # We can view B as two columns, each 4x1.
    # But Strassen works on 2x2 blocks.
    # Let's split A into two 2x2 blocks? No, A is 2x4.
    # Let's split B into two 2x2 blocks? No, B is 4x2.
    # B = [B_top; B_bottom] where B_top is 2x2, B_bottom is 2x2.
    # A = [A_left, A_right] where A_left is 2x2, A_right is 2x2.
    # C = A_left B_top + A_right B_bottom.
    # This is sum of two M_2,2,2.
    # So we can use two Strassen decompositions.
    
    # We need to map the indices for M_2,4,2.
    # A indices: i in {0,1}, j in {0,3}. a = i*4 + j.
    # B indices: j in {0,3}, k in {0,1}. b = j*2 + k.
    # C indices: k in {0,1}, i in {0,1}. c = k*2 + i.
    
    # Term 1 (A_left B_top):
    # A_left: i in {0,1}, j in {0,1}.
    # B_top: j in {0,1}, k in {0,1}.
    # C: k in {0,1}, i in {0,1}.
    # This is exactly M_2,2,2 but with different index mappings.
    
    # Let's create U24, V24, W24 for M_2,4,2.
    U24 = np.zeros((8, 14)) # 2*4 = 8
    V24 = np.zeros((8, 14)) # 4*2 = 8
    W24 = np.zeros((4, 14)) # 2*2 = 4
    
    # Block 1: A_left (j=0,1) * B_top (j=0,1)
    # Map Strassen U2 (4x7) to U24.
    # Strassen a' = i*2 + j'. Here j' is 0,1.
    # Actual a = i*4 + j'.
    # Strassen b' = j'*2 + k. Here j' is 0,1.
    # Actual b = j*2 + k.
    # Strassen c' = k*2 + i.
    # Actual c = k*2 + i.
    
    # So for r in 0..6:
    # U24[i*4 + j', r] = U2[i*2 + j', r]
    # V24[j*2 + k, r] = V2[j*2 + k, r]
    # W24[k*2 + i, r] = W2[k*2 + i, r]
    
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U24[i*4 + j, r] = U2[i*2 + j, r]
                
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V24[j*2 + k, r] = V2[j*2 + k, r]
                
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W24[k*2 + i, r] = W2[k*2 + i, r]
                
    # Block 2: A_right (j=2,3) * B_bottom (j=2,3)
    # A_right: i in {0,1}, j in {2,3}.
    # B_bottom: j in {2,3}, k in {0,1}.
    # C: same.
    
    # Map Strassen U2 to U24 with offset.
    # Strassen a' = i*2 + j'. Here j' is 0,1 corresponding to j=2,3.
    # Actual a = i*4 + (j'+2).
    # Strassen b' = j'*2 + k. Here j' is 0,1 corresponding to j=2,3.
    # Actual b = (j'+2)*2 + k.
    # Strassen c' = k*2 + i.
    # Actual c = k*2 + i.
    
    for i in range(2):
        for j in range(2):
            for r in range(7):
                U24[i*4 + (j+2), 7+r] = U2[i*2 + j, r]
                
    for j in range(2):
        for k in range(2):
            for r in range(7):
                V24[(j+2)*2 + k, 7+r] = V2[j*2 + k, r]
                
    for k in range(2):
        for i in range(2):
            for r in range(7):
                W24[k*2 + i, 7+r] = W2[k*2 + i, r]
                
    # Now U24, V24, W24 are for M_2,4,2 (top half of A).
    # We need M_4,4,2.
    # M_4,4,2 = M_2,4,2 (top A) + M_2,4,2 (bottom A).
    # Bottom A: i in {2,3}.
    # a = i*4 + j.
    # c = k*4 + i.
    
    U42 = np.zeros((16, 28))
    V42 = np.zeros((8, 28))
    W42 = np.zeros((8, 28)) # p*n = 2*4 = 8
    
    # Top A (i=0,1)
    for i in range(2):
        for j in range(4):
            for r in range(14):
                U42[i*4 + j, r] = U24[i*4 + j, r]
                
    for j in range(4):
        for k in range(2):
            for r in range(14):
                V42[j*2 + k, r] = V24[j*2 + k, r]
                
    for k in range(2):
        for i in range(2):
            for r in range(14):
                W42[k*4 + i, r] = W24[k*2 + i, r]
                
    # Bottom A (i=2,3)
    # Map i to i+2.
    for i in range(2):
        for j in range(4):
            for r in range(14):
                U42[(i+2)*4 + j, 14+r] = U24[i*4 + j, r]
                
    for j in range(4):
        for k in range(2):
            for r in range(14):
                V42[j*2 + k, 14+r] = V24[j*2 + k, r]
                
    for k in range(2):
        for i in range(2):
            for r in range(14):
                W42[k*4 + (i+2), 14+r] = W24[k*2 + i, r]
                
    # 3. M_4,4,1 part (rank 16)
    U41 = np.zeros((16, 16))
    V41 = np.zeros((4, 16)) # m*p = 4*1 = 4
    W41 = np.zeros((4, 16)) # p*n = 1*4 = 4
    
    col = 0
    for i in range(4):
        for j in range(4):
            U41[i*4 + j, col] = 1
            V41[j, col] = 1
            W41[i, col] = 1
            col += 1
            
    # Combine all parts
    # M_4,4,7 = M_4,4,4 + M_4,4,2 + M_4,4,1
    # Dimensions:
    # U: 16 x (49 + 28 + 16) = 16 x 93
    # V: 28 x (49 + 28 + 16)
    # W: 28 x (49 + 28 + 16)
    
    # Note: V for M_4,4,4 is 16x49? No.
    # M_4,4,4: V is (m*p) x r = 16 x 49.
    # M_4,4,2: V is (m*p) x r = 8 x 28.
    # M_4,4,1: V is (m*p) x r = 4 x 16.
    # Total V size: 28 x 93.
    # We need to place them correctly.
    # V for M_4,4,4 covers k=0..3. b = j*4 + k.
    # V for M_4,4,2 covers k=4,5. b = j*7 + k? No.
    # The tensor is direct sum.
    # For M_4,4,4, k is 0..3.
    # For M_4,4,2, k is 4,5.
    # For M_4,4,1, k is 6.
    
    # So V_44 has non-zeros only for b where k<4.
    # V_42 has non-zeros only for b where k in {4,5}.
    # V_41 has non-zeros only for b where k=6.
    
    U_total = np.zeros((16, 93))
    V_total = np.zeros((28, 93))
    W_total = np.zeros((28, 93))
    
    # Part 1: M_4,4,4
    U_total[:, 0:49] = U4
    # V4 is 16x49. V_total is 28x93.
    # V4 corresponds to k=0..3.
    # b = j*4 + k.
    # In V_total, b = j*7 + k.
    # So we need to map V4 indices to V_total indices.
    # V4 row index is j*4 + k (j in 0..3, k in 0..3).
    # V_total row index is j*7 + k.
    for j in range(4):
        for k in range(4):
            V_total[j*7 + k, 0:49] = V4[j*4 + k, 0:49]
            
    # W4 is 16x49. W_total is 28x93.
    # W4 corresponds to k=0..3.
    # c = k*4 + i.
    # In W_total, c = k*4 + i?
    # Wait, c = k*n + i. n=4.
    # For M_4,4,4, c = k*4 + i.
    # For M_4,4,7, c = k*4 + i.
    # So W4 maps directly to W_total rows 0..15.
    W_total[0:16, 0:49] = W4
    
    # Part 2: M_4,4,2
    U_total[:, 49:77] = U42
    # V42 is 8x28. Corresponds to k=4,5.
    # b = j*7 + k.
    # V42 row index is j*2 + k_local (k_local 0,1).
    # k = 4 + k_local.
    for j in range(4):
        for k_local in range(2):
            k = 4 + k_local
            V_total[j*7 + k, 49:77] = V42[j*2 + k_local, 0:28]
            
    # W42 is 8x28. Corresponds to k=4,5.
    # c = k*4 + i.
    # W42 row index is k_local*4 + i?
    # In M_4,4,2, c = k*2 + i? No.
    # In M_4,4,2, the output is 4x2.
    # c = k*4 + i.
    # W42 was constructed for M_4,4,2 with c = k*2 + i?
    # Let's re-verify W42 construction.
    # W24 was for M_2,4,2. c = k*2 + i.
    # W42 was built from W24.
    # W42 row index is k*4 + i?
    # In W42 construction:
    # W42[k*4 + i, r] = W24[k*2 + i, r].
    # So W42 row index is k*4 + i.
    # This matches c = k*4 + i for M_4,4,2.
    # So W42 maps to W_total rows 16..23 (k=4,5).
    W_total[16:24, 49:77] = W42
    
    # Part 3: M_4,4,1
    U_total[:, 77:93] = U41
    # V41 is 4x16. Corresponds to k=6.
    # b = j*7 + 6.
    # V41 row index is j.
    for j in range(4):
        V_total[j*7 + 6, 77:93] = V41[j, 0:16]
        
    # W41 is 4x16. Corresponds to k=6.
    # c = 6*4 + i = 24 + i.
    # W41 row index is i.
    W_total[24:28, 77:93] = W41
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
