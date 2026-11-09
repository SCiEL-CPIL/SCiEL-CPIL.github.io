import numpy as np

def construct():
    n, m, p = 2, 3, 5
    # We will build the decomposition piece by piece.
    # U shape: (n*m, rank) = (6, rank)
    # V shape: (m*p, rank) = (15, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_decomp(U_part, V_part, W_part, i_offset, j_offset, k_offset):
        """
        U_part: (n_sub * m_sub, r)
        V_part: (m_sub * p_sub, r)
        W_part: (p_sub * n_sub, r)
        i_offset: offset for i index in U and W
        j_offset: offset for j index in U and V
        k_offset: offset for k index in V and W
        """
        n_sub = U_part.shape[0] // m_sub
        m_sub = U_part.shape[0] // n_sub
        p_sub = W_part.shape[0] // n_sub
        
        r = U_part.shape[1]
        
        # Create full-sized matrices with zeros
        U_full = np.zeros((n*m, r))
        V_full = np.zeros((m*p, r))
        W_full = np.zeros((p*n, r))
        
        # Fill U
        for i in range(n_sub):
            for j in range(m_sub):
                row_u = (i + i_offset) * m + (j + j_offset)
                U_full[row_u, :] = U_part[i * m_sub + j, :]
                
        # Fill V
        for j in range(m_sub):
            for k in range(p_sub):
                row_v = (j + j_offset) * p + (k + k_offset)
                V_full[row_v, :] = V_part[j * p_sub + k, :]
                
        # Fill W
        for k in range(p_sub):
            for i in range(n_sub):
                row_w = (k + k_offset) * n + (i + i_offset)
                W_full[row_w, :] = W_part[k * n_sub + i, :]
                
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)

    # --- Part 1: 2x2x2 Strassen ---
    # A_2x2, B_2x2 -> C_2x2
    # U: 4x7, V: 4x7, W: 4x7
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # Indices for A: 00, 01, 10, 11
    # Indices for B: 00, 01, 10, 11
    # Indices for C: 00, 01, 10, 11
    
    # M1 = (A00 + A11)(B00 + B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[1, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = 1
    # C00 += M1, C01 += M1, C10 += M1, C11 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's use the standard Strassen formulas.
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Reset W_s to 0 and fill carefully.
    W_s[:] = 0
    
    # M1
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] += 1 # C00
    W_s[3, 0] += 1 # C11
    
    # M2
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] += 1 # C10
    W_s[3, 1] += -1 # C11
    
    # M3
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] += 1 # C01
    W_s[3, 2] += 1 # C11
    
    # M4
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] += 1 # C00
    W_s[2, 3] += 1 # C10
    
    # M5
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[1, 4] += 1 # C01
    W_s[0, 4] += -1 # C00
    
    # M6
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] += 1 # C11
    
    # M7
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] += 1 # C00
    
    # Check W_s:
    # C00: M1(1) + M4(1) - M5(-1) + M7(1) = 1+1-(-1)+1? No, M5 is -1 in C00.
    # My W_s[0,4] = -1. So sum is 1+1-1+1 = 2? No.
    # C00 = M1 + M4 - M5 + M7.
    # M1 coeff 1. M4 coeff 1. M5 coeff -1. M7 coeff 1.
    # My W_s[0,0]=1, W_s[0,3]=1, W_s[0,4]=-1, W_s[0,6]=1. Sum = 2?
    # Wait, Strassen C00 = M1 + M4 - M5 + M7.
    # Let's re-verify Strassen.
    # C00 = M1 + M4 - M5 + M7.
    # C01 = M3 + M5.
    # C10 = M2 + M4.
    # C11 = M1 - M2 + M3 + M6.
    
    # My W_s:
    # Col 0 (M1): C00=1, C11=1.
    # Col 1 (M2): C10=1, C11=-1.
    # Col 2 (M3): C01=1, C11=1.
    # Col 3 (M4): C00=1, C10=1.
    # Col 4 (M5): C01=1, C00=-1.
    # Col 5 (M6): C11=1.
    # Col 6 (M7): C00=1.
    
    # Sum for C00: 1 (M1) + 1 (M4) - 1 (M5) + 1 (M7) = 2. Error.
    # Strassen C00 = M1 + M4 - M5 + M7.
    # Ah, M5 is (A00+A01)B11.
    # C00 = M1 + M4 - M5 + M7.
    # If I set W_s[0,4] = -1, then sum is 1+1-1+1 = 2.
    # Wait, 1+1-1+1 = 2.
    # Is Strassen C00 = M1 + M4 - M5 + M7?
    # Yes.
    # Why do I get 2?
    # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # M4 = A11(B10-B00) = A11B10 - A11B00
    # M5 = (A00+A01)B11 = A00B11 + A01B11
    # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
    # Sum:
    # A00B00: 1
    # A00B11: 1 - 1 = 0
    # A11B00: 1 - 1 = 0
    # A11B11: 1 - 1 = 0
    # A11B10: 1 - 1 = 0
    # A01B11: -1 + 1 = 0
    # A01B10: 1
    # Result: A00B00 + A01B10.
    # Correct C00 = A00B00 + A01B10.
    # So the coefficients are correct. The sum of coefficients is 2, but the terms cancel out.
    # The decomposition is correct.

    add_decomp(U_s, V_s, W_s, 0, 0, 0)

    # --- Part 2: 2x2x1 Schoolbook ---
    # A_2x2, B_2x1 -> C_2x1
    # U: 4x2, V: 2x2, W: 2x2
    U_b1 = np.zeros((4, 2))
    V_b1 = np.zeros((2, 2))
    W_b1 = np.zeros((2, 2))
    
    # C00 = A00 B0
    U_b1[0, 0] = 1
    V_b1[0, 0] = 1
    W_b1[0, 0] = 1
    
    # C10 = A10 B0
    U_b1[2, 1] = 1
    V_b1[0, 1] = 1
    W_b1[1, 1] = 1
    
    add_decomp(U_b1, V_b1, W_b1, 0, 0, 2) # k offset 2 (3rd col of 3)

    # --- Part 3: 2x1x3 Schoolbook ---
    # A_2x1, B_1x3 -> C_2x3
    # U: 2x6, V: 3x6, W: 6x6
    U_b2 = np.zeros((2, 6))
    V_b2 = np.zeros((3, 6))
    W_b2 = np.zeros((6, 6))
    
    for i in range(2):
        for k in range(3):
            idx = i * 3 + k
            U_b2[i, idx] = 1
            V_b2[k, idx] = 1
            W_b2[k * 2 + i, idx] = 1
            
    add_decomp(U_b2, V_b2, W_b2, 0, 2, 0) # j offset 2 (3rd row of 3)

    # --- Part 4: 2x2x2 Strassen (for k=3,4) ---
    # Same as Part 1, but k offset 3
    add_decomp(U_s, V_s, W_s, 0, 0, 3)

    # --- Part 5: 2x1x2 Schoolbook ---
    # A_2x1, B_1x2 -> C_2x2
    # U: 2x4, V: 2x4, W: 4x4
    U_b3 = np.zeros((2, 4))
    V_b3 = np.zeros((2, 4))
    W_b3 = np.zeros((4, 4))
    
    for i in range(2):
        for k in range(2):
            idx = i * 2 + k
            U_b3[i, idx] = 1
            V_b3[k, idx] = 1
            W_b3[k * 2 + i, idx] = 1
            
    add_decomp(U_b3, V_b3, W_b3, 0, 2, 3) # j offset 2, k offset 3

    # Concatenate
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Convert to list of lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
