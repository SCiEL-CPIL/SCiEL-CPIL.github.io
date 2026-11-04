def construct():
    n, m, p = 2, 5, 6
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_tensor(u_vec, v_vec, w_vec):
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    # Helper to create zero vectors
    def zero_vec(dim):
        return [0] * dim

    # Strassen for 2x2 x 2x2
    # Local indices:
    # A: (i, j) i in 0..1, j in 0..1
    # B: (j, k) j in 0..1, k in 0..1
    # C: (i, k) i in 0..1, k in 0..1
    # Global mapping:
    # A_idx(i, j) = i * m + j_A_offset
    # B_idx(j, k) = (j_B_offset + j) * p + (k_B_offset + k)
    # C_idx(i, k) = (k_C_offset + k) * n + i
    
    def get_strassen_terms(j_A_offset, j_B_offset, k_B_offset, k_C_offset):
        terms = []
        
        # M1 = (A00 + A11) * (B00 + B11)
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[0*m + j_A_offset + 0] = 1
        u[1*m + j_A_offset + 1] = 1
        v[(j_B_offset + 0)*p + k_B_offset + 0] = 1
        v[(j_B_offset + 1)*p + k_B_offset + 1] = 1
        w[(k_C_offset + 0)*n + 0] = 1
        w[(k_C_offset + 0)*n + 1] = 1
        w[(k_C_offset + 1)*n + 0] = 1
        w[(k_C_offset + 1)*n + 1] = 1
        terms.append((u, v, w))
        
        # M2 = (A10 + A11) * B00
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[1*m + j_A_offset + 0] = 1
        u[1*m + j_A_offset + 1] = 1
        v[(j_B_offset + 0)*p + k_B_offset + 0] = 1
        w[(k_C_offset + 1)*n + 0] = 1
        w[(k_C_offset + 1)*n + 1] = 1
        terms.append((u, v, w))
        
        # M3 = A00 * (B10 - B11)
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[0*m + j_A_offset + 0] = 1
        v[(j_B_offset + 1)*p + k_B_offset + 0] = 1
        v[(j_B_offset + 1)*p + k_B_offset + 1] = -1
        w[(k_C_offset + 0)*n + 1] = 1
        w[(k_C_offset + 1)*n + 1] = 1
        terms.append((u, v, w))
        
        # M4 = A11 * (B01 - B00)
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[1*m + j_A_offset + 1] = 1
        v[(j_B_offset + 0)*p + k_B_offset + 1] = 1
        v[(j_B_offset + 0)*p + k_B_offset + 0] = -1
        w[(k_C_offset + 0)*n + 0] = 1
        w[(k_C_offset + 1)*n + 0] = 1
        terms.append((u, v, w))
        
        # M5 = (A00 + A01) * B11
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[0*m + j_A_offset + 0] = 1
        u[0*m + j_A_offset + 1] = 1
        v[(j_B_offset + 1)*p + k_B_offset + 1] = 1
        w[(k_C_offset + 0)*n + 0] = -1
        w[(k_C_offset + 1)*n + 1] = -1
        terms.append((u, v, w))
        
        # M6 = (A10 - A00) * (B00 + B01)
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[1*m + j_A_offset + 0] = 1
        u[0*m + j_A_offset + 0] = -1
        v[(j_B_offset + 0)*p + k_B_offset + 0] = 1
        v[(j_B_offset + 0)*p + k_B_offset + 1] = 1
        w[(k_C_offset + 1)*n + 1] = 1
        terms.append((u, v, w))
        
        # M7 = (A01 - A11) * (B10 + B11)
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        u[0*m + j_A_offset + 1] = 1
        u[1*m + j_A_offset + 1] = -1
        v[(j_B_offset + 1)*p + k_B_offset + 0] = 1
        v[(j_B_offset + 1)*p + k_B_offset + 1] = 1
        w[(k_C_offset + 0)*n + 0] = 1
        w[(k_C_offset + 1)*n + 1] = 1
        terms.append((u, v, w))
        
        return terms

    # Add Block 1: A[0:2, 0:2] * B[0:2, 0:6]
    # A cols 0,1. B rows 0,1.
    for t in range(3):
        k_off = 2 * t
        terms = get_strassen_terms(0, 0, k_off, k_off)
        for u, v, w in terms:
            add_tensor(u, v, w)

    # Add Block 2: A[0:2, 2:5] * B[2:5, 0:6]
    # A cols 2,3,4. B rows 2,3,4.
    for t in range(3):
        k_off = 2 * t
        # Part 2a: A[0:2, 2:4] * B[2:4, 2t:2t+2]
        # A cols 2,3. B rows 2,3.
        # Local j_A_offset = 2, j_B_offset = 2
        terms = get_strassen_terms(2, 2, k_off, k_off)
        for u, v, w in terms:
            add_tensor(u, v, w)
            
        # Part 2b: A[0:2, 4:5] * B[4:5, 2t:2t+2]
        # A col 4. B row 4.
        # This is 2x1 x 2. Rank 1.
        u = zero_vec(n*m); v = zero_vec(m*p); w = zero_vec(p*n)
        # A indices: i=0, j=4 -> 4; i=1, j=4 -> 9
        u[0*m + 4] = 1
        u[1*m + 4] = 1
        # B indices: j=4, k=k_off -> 4*6 + k_off; j=4, k=k_off+1 -> 4*6 + k_off + 1
        v[4*p + k_off] = 1
        v[4*p + k_off + 1] = 1
        # C indices: k=k_off, i=0 -> k_off*2 + 0; k=k_off, i=1 -> k_off*2 + 1; k=k_off+1, i=0 -> (k_off+1)*2 + 0; k=k_off+1, i=1 -> (k_off+1)*2 + 1
        w[k_off*n + 0] = 1
        w[k_off*n + 1] = 1
        w[(k_off+1)*n + 0] = 1
        w[(k_off+1)*n + 1] = 1
        add_tensor(u, v, w)

    U = [col for col in zip(*U_cols)] if U_cols else []
    V = [col for col in zip(*V_cols)] if V_cols else []
    W = [col for col in zip(*W_cols)] if W_cols else []
    
    # Convert to lists of lists for return
    U = [list(row) for row in zip(*U_cols)] if U_cols else []
    V = [list(row) for row in zip(*V_cols)] if V_cols else []
    W = [list(row) for row in zip(*W_cols)] if W_cols else []
    
    return {"U": U, "V": V, "W": W}
