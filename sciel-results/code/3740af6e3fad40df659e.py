def construct():
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: 2x1x10 (Tail)
    # A cols 2, B rows 2
    for i in range(2):
        for k in range(10):
            # U index: i * 3 + 2
            U[i * 3 + 2, col] = 1.0
            # V index: 2 * 10 + k
            V[2 * 10 + k, col] = 1.0
            # W index: k * 2 + i
            W[k * 2 + i, col] = 1.0
            col += 1
            
    # Part 2: 2x2x10 (Head)
    # A cols 0,1. B rows 0,1.
    # 5 blocks of 2x2x2
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 multiplications
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B01 - B11)
        # M4 = A11(B10 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # Coefficients for A (u)
        # Indices: 00, 01, 10, 11 -> global 0, 1, 3, 4
        # Note: A is 2x3.
        # A[0,0] -> U[0]
        # A[0,1] -> U[1]
        # A[1,0] -> U[3]
        # A[1,1] -> U[4]
        
        # Coefficients for B (v)
        # B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        # Global V indices:
        # B[0,k] -> V[0 * 10 + k]
        # B[1,k] -> V[1 * 10 + k]
        
        # Coefficients for C (w)
        # C[i,k]
        # Global W indices:
        # C[i,k] -> W[k * 2 + i]
        
        # M1
        U[0, col] = 1.0
        U[4, col] = 1.0
        V[k0, col] = 1.0
        V[10 + k1, col] = 1.0
        W[k0 * 2 + 0, col] = 1.0 # C[0,k0]
        W[k1 * 2 + 1, col] = 1.0 # C[1,k1]
        col += 1
        
        # M2
        U[3, col] = 1.0 # A10
        U[4, col] = 1.0 # A11
        V[k0, col] = 1.0 # B00
        W[k0 * 2 + 1, col] = 1.0 # C[1,k0]
        col += 1
        
        # M3
        U[0, col] = 1.0 # A00
        V[10 + k1, col] = -1.0 # B01 - B11? No, B01 is B[0,k1], B11 is B[1,k1]
        # Wait, M3 = A00 * (B01 - B11)
        # B01 is B[0, k1] -> V[k1]
        # B11 is B[1, k1] -> V[10 + k1]
        V[k1, col] = 1.0
        V[10 + k1, col] = -1.0
        W[k1 * 2 + 0, col] = 1.0 # C[0,k1]
        col += 1
        
        # M4
        U[4, col] = 1.0 # A11
        V[10 + k0, col] = 1.0 # B10
        V[k0, col] = -1.0 # B00
        W[k0 * 2 + 1, col] = 1.0 # C[1,k0]
        col += 1
        
        # M5
        U[0, col] = 1.0 # A00
        U[1, col] = 1.0 # A01
        V[10 + k1, col] = 1.0 # B11
        W[k1 * 2 + 0, col] = 1.0 # C[0,k1]
        col += 1
        
        # M6
        U[3, col] = 1.0 # A10
        U[0, col] = -1.0 # A00
        V[k0, col] = 1.0 # B00
        V[k1, col] = 1.0 # B01
        W[k1 * 2 + 1, col] = 1.0 # C[1,k1]
        col += 1
        
        # M7
        U[1, col] = 1.0 # A01
        U[4, col] = -1.0 # A11
        V[10 + k0, col] = 1.0 # B10
        V[10 + k1, col] = 1.0 # B11
        W[k0 * 2 + 0, col] = 1.0 # C[0,k0]
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
