def _build_block_strassen():
    n, m, p = 5, 5, 5
    # Max rank estimate: 120
    max_rank = 120
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    rank = 0
    
    # Helper to add a schoolbook block
    def add_schoolbook(i0, j0, k0, ri, cj, rk):
        # A block: rows i0..i0+ri-1, cols j0..j0+cj-1
        # B block: rows j0..j0+cj-1, cols k0..k0+rk-1
        # C block: rows i0..i0+ri-1, cols k0..k0+rk-1
        nonlocal rank
        for ii in range(ri):
            for jj in range(cj):
                for kk in range(rk):
                    u_idx = (i0 + ii) * m + (j0 + jj)
                    v_idx = (j0 + jj) * p + (k0 + kk)
                    w_idx = (k0 + kk) * n + (i0 + ii)
                    
                    U[u_idx, rank] = 1.0
                    V[v_idx, rank] = 1.0
                    W[w_idx, rank] = 1.0
                    rank += 1
                    
    # Helper to add a Strassen block (2x2)
    def add_strassen(i0, j0, k0):
        # A block: rows i0, i0+1; cols j0, j0+1
        # B block: rows j0, j0+1; cols k0, k0+1
        # C block: rows i0, i0+1; cols k0, k0+1
        nonlocal rank
        # Strassen indices
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        
        # M1
        U[i0*m + j0, rank] = 1; U[(i0+1)*m + (j0+1), rank] = 1
        V[j0*p + k0, rank] = 1; V[(j0+1)*p + (k0+1), rank] = 1
        W[k0*n + i0, rank] = 1; W[(k0+1)*n + (i0+1), rank] = 1
        rank += 1
        
        # M2
        U[(i0+1)*m + j0, rank] = 1; U[(i0+1)*m + (j0+1), rank] = 1
        V[j0*p + k0, rank] = 1
        W[k0*n + (i0+1), rank] = 1; W[(k0+1)*n + (i0+1), rank] = -1
        rank += 1
        
        # M3
        U[i0*m + j0, rank] = 1
        V[j0*p + (k0+1), rank] = 1; V[(j0+1)*p + (k0+1), rank] = -1
        W[(k0+1)*n + i0, rank] = 1; W[(k0+1)*n + (i0+1), rank] = 1
        rank += 1
        
        # M4
        U[(i0+1)*m + (j0+1), rank] = 1
        V[(j0+1)*p + k0, rank] = 1; V[j0*p + k0, rank] = -1
        W[k0*n + i0, rank] = 1; W[k0*n + (i0+1), rank] = 1
        rank += 1
        
        # M5
        U[i0*m + j0, rank] = 1; U[i0*m + (j0+1), rank] = 1
        V[(j0+1)*p + (k0+1), rank] = 1
        W[k0*n + i0, rank] = -1; W[(k0+1)*n + i0, rank] = 1
        rank += 1
        
        # M6
        U[(i0+1)*m + j0, rank] = 1; U[i0*m + j0, rank] = -1
        V[j0*p + k0, rank] = 1; V[j0*p + (k0+1), rank] = 1
        W[(k0+1)*n + (i0+1), rank] = 1
        rank += 1
        
        # M7
        U[i0*m + (j0+1), rank] = 1; U[(i0+1)*m + (j0+1), rank] = -1
        V[(j0+1)*p + k0, rank] = 1; V[(j0+1)*p + (k0+1), rank] = 1
        W[k0*n + i0, rank] = 1
        rank += 1

    # Block structure: 2, 2, 1
    # A blocks:
    # A11 (2x2) at (0,0)
    # A12 (2x2) at (0,2)
    # A13 (2x1) at (0,4)
    # A21 (2x2) at (2,0)
    # A22 (2x2) at (2,2)
    # A23 (2x1) at (2,4)
    # A31 (1x2) at (4,0)
    # A32 (1x2) at (4,2)
    # A33 (1x1) at (4,4)
    
    # C11 (2x2) = A11*B11 + A12*B21 + A13*B31
    add_strassen(0, 0, 0) # A11*B11
    add_strassen(0, 2, 0) # A12*B21 (A cols 2-3, B rows 2-3? No. A12 is rows 0-1, cols 2-3. B21 is rows 2-3, cols 0-1. 
                          # Wait. A12 (2x2) * B21 (2x2).
                          # A12 indices: i in 0..1, j in 2..3.
                          # B21 indices: j in 2..3, k in 0..1.
                          # My add_strassen assumes contiguous blocks starting at i0, j0, k0.
                          # For A12*B21:
                          # A block starts at i=0, j=2.
                          # B block starts at j=2, k=0.
                          # C block starts at i=0, k=0.
                          # So add_strassen(0, 2, 0) is correct for the mapping of indices.
    add_schoolbook(0, 4, 4, 2, 1, 2) # A13 (2x1) * B31 (1x2). A: i=0..1, j=4. B: j=4, k=4..5? No B31 is rows 4, cols 0..1.
    # Correction: B31 is rows 4 (j=4), cols 0..1 (k=0..1).
    # So add_schoolbook(i0=0, j0=4, k0=0, ri=2, cj=1, rk=2)
    # Wait, my add_schoolbook signature: (i0, j0, k0, ri, cj, rk)
    # A block: rows i0..i0+ri-1, cols j0..j0+cj-1.
    # B block: rows j0..j0+cj-1, cols k0..k0+rk-1.
    # C block: rows i0..i0+ri-1, cols k0..k0+rk-1.
    # For A13*B31:
    # A: i=0..1, j=4. (ri=2, cj=1)
    # B: j=4, k=0..1. (rk=2)
    # C: i=0..1, k=0..1.
    # So add_schoolbook(0, 4, 0, 2, 1, 2)
    
    # Let's re-verify the block indices for the 5x5 case.
    # Rows/Cols 0,1,2,3,4.
    # Split: 2, 2, 1.
    # Indices:
    # Group 0: 0, 1
    # Group 1: 2, 3
    # Group 2: 4
    
    # A11: i in G0, j in G0. (0,0)
    # A12: i in G0, j in G1. (0,2)
    # A13: i in G0, j in G2. (0,4)
    # A21: i in G1, j in G0. (2,0)
    # A22: i in G1, j in G1. (2,2)
    # A23: i in G1, j in G2. (2,4)
    # A31: i in G2, j in G0. (4,0)
    # A32: i in G2, j in G1. (4,2)
    # A33: i in G2, j in G2. (4,4)
    
    # B11: j in G0, k in G0. (0,0)
    # B12: j in G0, k in G1. (0,2)
    # B13: j in G0, k in G2. (0,4)
    # B21: j in G1, k in G0. (2,0)
    # B22: j in G1, k in G1. (2,2)
    # B23: j in G1, k in G2. (2,4)
    # B31: j in G2, k in G0. (4,0)
    # B32: j in G2, k in G1. (4,2)
    # B33: j in G2, k in G2. (4,4)
    
    # C11: i in G0, k in G0. (0,0)
    # C12: i in G0, k in G1. (0,2)
    # C13: i in G0, k in G2. (0,4)
    # C21: i in G1, k in G0. (2,0)
    # C22: i in G1, k in G1. (2,2)
    # C23: i in G1, k in G2. (2,4)
    # C31: i in G2, k in G0. (4,0)
    # C32: i in G2, k in G1. (4,2)
    # C33: i in G2, k in G2. (4,4)
    
    # Calculations:
    # C11 = A11*B11 + A12*B21 + A13*B31
    # A11(0,0) 2x2, B11(0,0) 2x2 -> Strassen(0,0,0)
    # A12(0,2) 2x2, B21(2,0) 2x2 -> Strassen(0,2,0)
    # A13(0,4) 2x1, B31(4,0) 1x2 -> Schoolbook(0,4,0, 2,1,2)
    
    # C12 = A11*B12 + A12*B22 + A13*B32
    # A11(0,0) 2x2, B12(0,2) 2x2 -> Strassen(0,0,2)
    # A12(0,2) 2x2, B22(2,2) 2x2 -> Strassen(0,2,2)
    # A13(0,4) 2x1, B32(4,2) 1x2 -> Schoolbook(0,4,2, 2,1,2)
    
    # C13 = A11*B13 + A12*B23 + A13*B33
    # A11(0,0) 2x2, B13(0,4) 2x1 -> Schoolbook(0,0,4, 2,2,1)
    # A12(0,2) 2x2, B23(2,4) 2x1 -> Schoolbook(0,2,4, 2,2,1)
    # A13(0,4) 2x1, B33(4,4) 1x1 -> Schoolbook(0,4,4, 2,1,1)
    
    # C21 = A21*B11 + A22*B21 + A23*B31
    # A21(2,0) 2x2, B11(0,0) 2x2 -> Strassen(2,0,0)
    # A22(2,2) 2x2, B21(2,0) 2x2 -> Strassen(2,2,0)
    # A23(2,4) 2x1, B31(4,0) 1x2 -> Schoolbook(2,4,0, 2,1,2)
    
    # C22 = A21*B12 + A22*B22 + A23*B32
    # A21(2,0) 2x2, B12(0,2) 2x2 -> Strassen(2,0,2)
    # A22(2,2) 2x2, B22(2,2) 2x2 -> Strassen(2,2,2)
    # A23(2,4) 2x1, B32(4,2) 1x2 -> Schoolbook(2,4,2, 2,1,2)
    
    # C23 = A21*B13 + A22*B23 + A23*B33
    # A21(2,0) 2x2, B13(0,4) 2x1 -> Schoolbook(2,0,4, 2,2,1)
    # A22(2,2) 2x2, B23(2,4) 2x1 -> Schoolbook(2,2,4, 2,2,1)
    # A23(2,4) 2x1, B33(4,4) 1x1 -> Schoolbook(2,4,4, 2,1,1)
    
    # C31 = A31*B11 + A32*B21 + A33*B31
    # A31(4,0) 1x2, B11(0,0) 2x2 -> Schoolbook(4,0,0, 1,2,2)
    # A32(4,2) 1x2, B21(2,0) 2x2 -> Schoolbook(4,2,0, 1,2,2)
    # A33(4,4) 1x1, B31(4,0) 1x2 -> Schoolbook(4,4,0, 1,1,2)
    
    # C32 = A31*B12 + A32*B22 + A33*B32
    # A31(4,0) 1x2, B12(0,2) 2x2 -> Schoolbook(4,0,2, 1,2,2)
    # A32(4,2) 1x2, B22(2,2) 2x2 -> Schoolbook(4,2,2, 1,2,2)
    # A33(4,4) 1x1, B32(4,2) 1x2 -> Schoolbook(4,4,2, 1,1,2)
    
    # C33 = A31*B13 + A32*B23 + A33*B33
    # A31(4,0) 1x2, B13(0,4) 2x1 -> Schoolbook(4,0,4, 1,2,1)
    # A32(4,2) 1x2, B23(2,4) 2x1 -> Schoolbook(4,2,4, 1,2,1)
    # A33(4,4) 1x1, B33(4,4) 1x1 -> Schoolbook(4,4,4, 1,1,1)

    # Execute the plan
    # C11
    add_strassen(0, 0, 0)
    add_strassen(0, 2, 0)
    add_schoolbook(0, 4, 0, 2, 1, 2)
    
    # C12
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 4, 2, 2, 1, 2)
    
    # C13
    add_schoolbook(0, 0, 4, 2, 2, 1)
    add_schoolbook(0, 2, 4, 2, 2, 1)
    add_schoolbook(0, 4, 4, 2, 1, 1)
    
    # C21
    add_strassen(2, 0, 0)
    add_strassen(2, 2, 0)
    add_schoolbook(2, 4, 0, 2, 1, 2)
    
    # C22
    add_strassen(2, 0, 2)
    add_strassen(2, 2, 2)
    add_schoolbook(2, 4, 2, 2, 1, 2)
    
    # C23
    add_schoolbook(2, 0, 4, 2, 2, 1)
    add_schoolbook(2, 2, 4, 2, 2, 1)
    add_schoolbook(2, 4, 4, 2, 1, 1)
    
    # C31
    add_schoolbook(4, 0, 0, 1, 2, 2)
    add_schoolbook(4, 2, 0, 1, 2, 2)
    add_schoolbook(4, 4, 0, 1, 1, 2)
    
    # C32
    add_schoolbook(4, 0, 2, 1, 2, 2)
    add_schoolbook(4, 2, 2, 1, 2, 2)
    add_schoolbook(4, 4, 2, 1, 1, 2)
    
    # C33
    add_schoolbook(4, 0, 4, 1, 2, 1)
    add_schoolbook(4, 2, 4, 1, 2, 1)
    add_schoolbook(4, 4, 4, 1, 1, 1)
    
    # Trim arrays
    U = U[:, :rank]
    V = V[:, :rank]
    W = W[:, :rank]
    
    return {'U': U, 'V': V, 'W': W}
