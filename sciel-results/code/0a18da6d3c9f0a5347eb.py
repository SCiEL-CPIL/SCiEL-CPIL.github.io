def construct():
    n, m, p = 2, 3, 8
    R = 44
    U = np.zeros((n*m, R), dtype=int)
    V = np.zeros((m*p, R), dtype=int)
    W = np.zeros((p*n, R), dtype=int)
    
    # Part 1: 2x1x8 block (A[:, 2] * B[2, :])
    # A index for col 2: i*3 + 2
    # B index for row 2: 2*8 + k = 16 + k
    # C index: k*2 + i
    r = 0
    for i in range(2):
        for k in range(8):
            A_idx = i * 3 + 2
            B_idx = 2 * 8 + k
            C_idx = k * 2 + i
            U[A_idx, r] = 1
            V[B_idx, r] = 1
            W[C_idx, r] = 1
            r += 1
            
    # Part 2: 2x2x8 block (A[:, 0:2] * B[0:2, :])
    # Split into 4 blocks of 2x2x2
    # Strassen indices for A (2x2):
    # A00 -> 0, A01 -> 1, A10 -> 3, A11 -> 4
    # Strassen indices for B (2x2) local:
    # B00, B01, B10, B11
    # Global B indices for block b:
    # B00 -> 0*8 + 2b = 2b
    # B01 -> 0*8 + 2b+1 = 2b+1
    # B10 -> 1*8 + 2b = 8+2b
    # B11 -> 1*8 + 2b+1 = 8+2b+1
    # Global C indices for block b:
    # C00 -> (2b)*2 + 0 = 4b
    # C01 -> (2b+1)*2 + 0 = 4b+2
    # C10 -> (2b)*2 + 1 = 4b+1
    # C11 -> (2b+1)*2 + 1 = 4b+3
    
    for b in range(4):
        # Local B indices
        b00 = 2*b
        b01 = 2*b + 1
        b10 = 8 + 2*b
        b11 = 8 + 2*b + 1
        
        # Local C indices
        c00 = 4*b
        c01 = 4*b + 2
        c10 = 4*b + 1
        c11 = 4*b + 3
        
        # A indices
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # Strassen M1 = (A00 + A11)(B00 + B11)
        U[a00, r] += 1
        U[a11, r] += 1
        V[b00, r] += 1
        V[b11, r] += 1
        W[c00, r] += 1
        W[c11, r] += 1
        r += 1
        
        # Strassen M2 = (A10 + A11) B00
        U[a10, r] += 1
        U[a11, r] += 1
        V[b00, r] += 1
        W[c10, r] += 1
        W[c11, r] -= 1
        r += 1
        
        # Strassen M3 = A00 (B01 - B11)
        U[a00, r] += 1
        V[b01, r] += 1
        V[b11, r] -= 1
        W[c01, r] += 1
        W[c11, r] += 1
        r += 1
        
        # Strassen M4 = A11 (B10 - B00)
        U[a11, r] += 1
        V[b10, r] += 1
        V[b00, r] -= 1
        W[c00, r] += 1
        W[c10, r] += 1
        r += 1
        
        # Strassen M5 = (A00 + A01) B11
        U[a00, r] += 1
        U[a01, r] += 1
        V[b11, r] += 1
        W[c00, r] -= 1
        W[c01, r] += 1
        r += 1
        
        # Strassen M6 = (A10 - A00)(B00 + B01)
        U[a10, r] += 1
        U[a00, r] -= 1
        V[b00, r] += 1
        V[b01, r] += 1
        W[c11, r] += 1
        r += 1
        
        # Strassen M7 = (A01 - A11)(B10 + B11)
        U[a01, r] += 1
        U[a11, r] -= 1
        V[b10, r] += 1
        V[b11, r] += 1
        W[c00, r] += 1
        W[c11, r] += 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
