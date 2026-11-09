import time

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # We'll build rank-54 decomposition:
    # Split A (2x5) into A0(2x2), A1(2x2), A2(2x1)
    # Split B (5x6) into B0(2x6), B1(2x6), B2(1x6)
    # C = A0@B0 + A1@B1 + A2@B2
    # A0@B0: split B0 into 3 blocks of 2x2, each done via Strassen (rank 7 each) = 21
    # A1@B1: same = 21
    # A2@B2: outer product, rank 12
    # Total rank: 54
    
    U_terms = []  # list of (row_index, value)
    V_terms = []  # list of (row_index, value)
    W_terms = []  # list of (row_index, value)
    
    col = 0
    
    def add_term(u_row, v_row, w_row, val):
        nonlocal col
        U_terms.append((u_row, val))
        V_terms.append((v_row, val))
        W_terms.append((w_row, val))
        col += 1
    
    # Strassen rank-1 terms for 2x2@2x2
    # P = [[a,b],[c,d]], Q = [[e,f],[g,h]], R = P@Q
    # M1 = (a+d)*(e+h): U entries: a:+1, d:+1; V: e:+1, h:+1; W: R00:+1, R11:+1
    # Actually, each Mj contributes to specific R entries.
    # Let me define the 7 Strassen bilinear forms and their contributions to R.
    
    # P indices: a=P[0,0], b=P[0,1], c=P[1,0], d=P[1,1]
    # Q indices: e=Q[0,0], f=Q[0,1], g=Q[1,0], h=Q[1,1]
    # R = P@Q: R00, R01, R10, R11
    
    # The 7 Strassen multipliers and their contributions to R:
    # M1 = (a+d)(e+h) -> R00: +1, R11: +1
    # M2 = (c+d)(e)    -> R10: +1
    # M3 = (a)(f-h)    -> R01: +1, R11: +1  ... wait let me re-derive
    
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 - M2 + M3 + M4
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M4 + M7  -- wait, let me get this right
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each Mj is a product of a linear form in A and a linear form in B.
    # So for each Mj, we have one rank-1 term in the tensor decomposition.
    
    # For the full matrix mult tensor, a rank-1 term has:
    # U[i*m+j] = coefficient for A[i,j]
    # V[j*p+k] = coefficient for B[j,k]  -- wait, V index is j*p+k where j is row of B, k is col
    # Actually, B is m x p, so B[j,k] with j in 0..m-1, k in 0..p-1.
    # V row index: j*p + k
    # W row index: k*n + i (output C[i,k])
    
    # So for a 2x2@2x2 subproblem embedded at positions:
    # A rows: i0, i0+1 (2 rows)
    # A cols (which are B rows): j0, j0+1
    # B rows: j0, j0+1
    # B cols: k0, k0+1
    # Output C rows: i0, i0+1
    # Output C cols: k0, k0+1
    
    # For Strassen on A_block (2x2) @ B_block (2x2) -> C_block (2x2):
    # A_block[A_r, A_c] corresponds to global A[i0+A_r, j0+A_c], index (i0+A_r)*m + (j0+A_c)
    # B_block[B_r, B_c] corresponds to global B[j0+B_r, k0+B_c], index (j0+B_r)*p + (k0+B_c)
    # C_block[C_r, C_c] corresponds to global C[i0+C_r, k0+C_c], index (k0+C_c)*n + (i0+C_r)
    
    def add_strassen_2x2(i0, j0, k0):
        """Add 7 Strassen rank-1 terms for 2x2@2x2 at A rows i0,i0+1; B rows j0,j0+1; C cols k0,k0+1."""
        nonlocal col
        
        # Helper to add a single rank-1 term with given A-coeff, B-coeff, C-coeff
        # A-coeff: dict mapping (ar, ac) -> val
        # B-coeff: dict mapping (br, bc) -> val
        # C-coeff: dict mapping (cr, cc) -> val
        def add_one(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            for (ar, ac), aval in a_coeffs.items():
                u_row = (i0 + ar) * m + (j0 + ac)
                U_terms.append((u_row, aval))
            for (br, bc), bval in b_coeffs.items():
                v_row = (j0 + br) * p + (k0 + bc)
                V_terms.append((v_row, bval))
            for (cr, cc), cval in c_coeffs.items():
                w_row = (k0 + cc) * n + (i0 + cr)
                W_terms.append((w_row, cval))
            col += 1
        
        # M1 = (A00+A11)(B00+B11) -> C00:+1, C11:+1
        a = {(0,0):1, (1,1):1}
        b = {(0,0):1, (1,1):1}
        c = {(0,0):1, (1,1):1}
        add_one(a, b, c)
        
        # M2 = (A10+A11)B00 -> C10:+1
        a = {(1,0):1, (1,1):1}
        b = {(0,0):1}
        c = {(1,0):1}
        add_one(a, b, c)
        
        # M3 = A00(B01-B11) -> C01:+1, C11:+1
        a = {(0,0):1}
        b = {(0,1):1, (1,1):-1}
        c = {(0,1):1, (1,1):1}
        add_one(a, b, c)
        
        # M4 = A11(B10-B00) -> C00:+1, C10:+1
        a = {(1,1):1}
        b = {(1,0):1, (0,0):-1}
        c = {(0,0):1, (1,0):1}
        add_one(a, b, c)
        
        # M5 = (A00+A01)B11 -> C00:-1, C01:+1
        a = {(0,0):1, (0,1):1}
        b = {(1,1):1}
        c = {(0,0):-1, (0,1):1}
        add_one(a, b, c)
        
        # M6 = (A10-A00)(B00+B01) -> C11:+1
        a = {(1,0):1, (0,0):-1}
        b = {(0,0):1, (0,1):1}
        c = {(1,1):1}
        add_one(a, b, c)
        
        # M7 = (A01-A11)(B10+B11) -> C00:+1
        a = {(0,1):1, (1,1):-1}
        b = {(1,0):1, (1,1):1}
        c = {(0,0):1}
        add_one(a, b, c)
    
    # Now assemble:
    # Part 1: A0(2x2) @ B0(2x6)
    # A0: A rows 0,1, cols 0,1 -> i0=0, j0=0
    # B0: B rows 0,1, cols 0..5 -> split into 3 blocks of 2 cols
    # B0a: cols 0,1 -> k0=0
    # B0b: cols 2,3 -> k0=2
    # B0c: cols 4,5 -> k0=4
    for k0 in [0, 2, 4]:
        add_strassen_2x2(0, 0, k0)
    
    # Part 2: A1(2x2) @ B1(2x6)
    # A1: A rows 0,1, cols 2,3 -> i0=0, j0=2
    # B1: B rows 2,3, cols 0..5 -> split into 3 blocks
    for k0 in [0, 2, 4]:
        add_strassen_2x2(0, 2, k0)
    
    # Part 3: A2(2x1) @ B2(1x6)
    # A2: A rows 0,1, col 4 -> j=4
    # B2: B row 4, cols 0..5
    # C[i,k] += A[i,4] * B[4,k]
    for i in range(2):
        for k in range(6):
            u_row = i * m + 4
            v_row = 4 * p + k
            w_row = k * n + i
            U_terms.append((u_row, 1))
            V_terms.append((v_row, 1))
            W_terms.append((w_row, 1))
            col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, val) in U_terms:
        U[row][col] = val  # Wait, col has been incremented...
    
    # Let me redo this more carefully.
    U_list = [[] for _ in range(n * m)]
    V_list = [[] for _ in range(m * p)]
    W_list = [[] for _ in range(p * n)]
    
    # Rebuild
    col = 0
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_strassen_2x2_v2(i0, j0, k0):
        nonlocal col
        def add_one(a_coeffs, b_coeffs, c_coeffs):
            nonlocal col
            u_dict = {}
            for (ar, ac), aval in a_coeffs.items():
                u_row = (i0 + ar) * m + (j0 + ac)
                u_dict[u_row] = aval
            v_dict = {}
            for (br, bc), bval in b_coeffs.items():
                v_row = (j0 + br) * p + (k0 + bc)
                v_dict[v_row] = bval
            w_dict = {}
            for (cr, cc), cval in c_coeffs.items():
                w_row = (k0 + cc) * n + (i0 + cr)
                w_dict[w_row] = cval
            U_terms.append(u_dict)
            V_terms.append(v_dict)
            W_terms.append(w_dict)
            col += 1
        
        add_one({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1})
        add_one({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1})
        add_one({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1})
        add_one({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1})
        add_one({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1})
        add_one({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1})
        add_one({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1})
    
    # Part 1: A0@B0
    for k0 in [0, 2, 4]:
        add_strassen_2x2_v2(0, 0, k0)
    
    # Part 2: A1@B1
    for k0 in [0, 2, 4]:
        add_strassen_2x2_v2(0, 2, k0)
    
    # Part 3: A2@B2
    for i in range(2):
        for k in range(6):
            U_terms.append({i * m + 4: 1})
            V_terms.append({4 * p + k: 1})
            W_terms.append({k * n + i: 1})
            col += 1
    
    rank = col
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for ci in range(rank):
        for row, val in U_terms[ci].items():
            U[row][ci] = val
        for row, val in V_terms[ci].items():
            V[row][ci] = val
        for row, val in W_terms[ci].items():
            W[row][ci] = val
    
    return {"U": U, "V": V, "W": W}
