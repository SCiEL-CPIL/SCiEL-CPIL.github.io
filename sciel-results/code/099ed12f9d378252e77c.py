def construct():
    n, m, p = 3, 4, 4
    # We'll build U (12 x rank), V (16 x rank), W (12 x rank)
    # Split: row 0 (1x4) handled naively, rows 1-2 (2x4) handled with Strassen
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # Part 1: A[0,:] (1x4) * B (4x4) -> C[0,:] (1x4)
    # Naive: 16 terms
    for j in range(4):
        for k in range(4):
            # U: row index = 0*4 + j = j (in the 1x4 block, which is rows 0-3 of U)
            # V: row index = j*4 + k
            # W: row index = k*3 + 0 = k (in the 1x4 output, rows 0-3 of W)
            U_rows.append((j, 1))
            V_rows.append((j * 4 + k, 1))
            W_rows.append((k * 3 + 0, 1))
    
    # Part 2: A[1:3,:] (2x4) * B (4x4) -> C[1:3,:] (2x4)
    # Split inner j into 0-1 and 2-3, output k into 0-1 and 2-3
    # 4 sub-problems, each 2x2x2, using Strassen (rank 7 each)
    
    def strassen_2x2x2(U_offset, V_offset, W_offset, i0, j0, k0):
        """Add Strassen rank-7 terms for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        # Strassen 7 products:
        # M1 = (A00+A11)(B01+B10)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A10)B11
        # M6 = (A11-A00)(B00+B11)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1+M4-M5+M7
        # C01 = M1+M5
        # C10 = M2+M4
        # C11 = M1-M2+M3+M6
        
        terms = []
        # M1 = (A00+A11)(B01+B10)
        terms.append(([(i0,j0),(i0+1,j0+1)], [(j0,k0+1),(j0+1,k0)], [(i0,k0),(i0,k0+1),(i0+1,k0),(i0+1,k0+1)], [1,1,1,1]))
        # M2 = (A10+A11)B00
        terms.append(([(i0+1,j0),(i0+1,j0+1)], [(j0,k0)], [(i0+1,k0)], [1,1,1]))
        # M3 = A00(B10-B11)
        terms.append(([(i0,j0)], [(j0+1,k0),(j0+1,k0+1)], [(i0+1,k0),(i0+1,k0+1)], [1,-1,1,-1]))
        # M4 = A11(B00-B10)
        terms.append(([(i0+1,j0+1)], [(j0,k0),(j0+1,k0)], [(i0,k0),(i0+1,k0)], [1,-1,1,-1]))
        # M5 = (A00+A10)B11
        terms.append(([(i0,j0),(i0+1,j0)], [(j0+1,k0+1)], [(i0,k0+1),(i0+1,k0+1)], [1,1,1,1]))
        # M6 = (A11-A00)(B00+B11)
        terms.append(([(i0+1,j0+1),(i0,j0)], [(j0,k0),(j0+1,k0+1)], [(i0+1,k0+1)], [1,-1,1,-1]))
        # M7 = (A01-A11)(B10+B11)
        terms.append(([(i0,j0+1),(i0+1,j0+1)], [(j0+1,k0),(j0+1,k0+1)], [(i0,k0)], [1,-1,1,-1]))
        
        for u_entries, v_entries, w_entries, w_coeffs in terms:
            for (ui, uj), (vi, vk), (wi, wk), wc in zip(u_entries, v_entries, w_entries, w_coeffs):
                U_rows.append((ui * 4 + uj, 1))
                V_rows.append((vi * 4 + vk, 1))
                W_rows.append((wk * 3 + wi, wc))
    
    # Sub-problem 1: i=1,2; j=0,1; k=0,1
    strassen_2x2x2(0, 0, 0, i0=1, j0=0, k0=0)
    # Sub-problem 2: i=1,2; j=0,1; k=2,3
    strassen_2x2x2(0, 0, 0, i0=1, j0=0, k0=2)
    # Sub-problem 3: i=1,2; j=2,3; k=0,1
    strassen_2x2x2(0, 0, 0, i0=1, j0=2, k0=0)
    # Sub-problem 4: i=1,2; j=2,3; k=2,3
    strassen_2x2x2(0, 0, 0, i0=1, j0=2, k0=2)
    
    rank = len(U_rows)
    U = [[0]*rank for _ in range(12)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(12)]
    
    for col, (u_idx, u_val) in enumerate(U_rows):
        U[u_idx][col] = u_val
    for col, (v_idx, v_val) in enumerate(V_rows):
        V[v_idx][col] = v_val
    for col, (w_idx, w_val) in enumerate(W_rows):
        W[w_idx][col] = w_val
    
    return {"U": U, "V": V, "W": W}
