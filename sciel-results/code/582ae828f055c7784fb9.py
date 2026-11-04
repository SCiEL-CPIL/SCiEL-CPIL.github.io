"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # 1. A_R B_B (Rank 10)
    # A_R is col 2 of A. B_B is row 2 of B.
    # C[i, k] += A[i, 2] * B[2, k]
    for i in range(2):
        for k in range(5):
            u_idx = i * 3 + 2
            v_idx = 2 * 5 + k
            w_idx = k * 2 + i
            add_term(u_idx, v_idx, w_idx, 1)
            
    # 2. A_L B_T1 (Rank 7)
    # A_L is cols 0,1. B_T1 is rows 0,1, cols 0,1.
    # Strassen
    # M1 = (a+d)(e+h)
    add_term(0*3+0, 0*5+0, 0*2+0, 1) # a*e -> C00
    add_term(0*3+0, 0*5+1, 0*2+0, 1) # a*h -> C00 (Wait, M1 is (a+d)(e+h) = ae+ah+de+dh)
    # My previous manual mapping was for the *multiplications* M1..M7.
    # Let's stick to the explicit M1..M7 definitions.
    
    # Reset col to 10
    col = 10
    
    # M1: (a+d)(e+h)
    # U: a(0,0), d(1,1). V: e(0,0), h(1,1).
    # This requires 4 entries in U, 4 in V. But it's ONE multiplication column.
    # The column corresponds to the product of linear forms.
    # U vector: [1, 0, 0, 1, 0, 0] (a, b, c, d, ...)
    # V vector: [1, 0, 0, 1, 0, 0, ...]
    # W vector: adds to C00 and C11.
    
    # Let's define a function to add a Strassen block for specific k-offset
    def add_strassen_block(k_offset):
        nonlocal col
        # M1: (a+d)(e+h)
        u_vec = [0]*6; u_vec[0]=1; u_vec[3]=1
        v_vec = [0]*15; v_vec[0*5+k_offset]=1; v_vec[1*5+(k_offset+1)]=1
        w_vec = [0]*10
        # C00 (k=k_offset, i=0) -> idx k_offset*2+0
        w_vec[k_offset*2+0] += 1
        # C11 (k=k_offset+1, i=1) -> idx (k_offset+1)*2+1
        w_vec[(k_offset+1)*2+1] += 1
        
        # We need to add this as a single column with coefficients?
        # No, the decomposition is sum of rank-1 tensors.
        # M1 is a product of two linear forms. It contributes to multiple W entries.
        # So for this column:
        U_col = u_vec
        V_col = v_vec
        W_col = w_vec
        
        for r in range(6):
            if U_col[r] != 0: U[r][col] = U_col[r]
        for r in range(15):
            if V_col[r] != 0: V[r][col] = V_col[r]
        for r in range(10):
            if W_col[r] != 0: W[r][col] = W_col[r]
        col += 1

        # M2: (c+d)e
        u_vec = [0]*6; u_vec[3]=1; u_vec[4]=1 # c(1,0)=3, d(1,1)=4? No.
        # A indices: i*3+j.
        # a=0,0 (0). b=0,1 (1). c=1,0 (3). d=1,1 (4).
        # M2: (c+d) e. c is idx 3, d is idx 4. e is B[0,0] (idx k_offset).
        u_vec = [0]*6; u_vec[3]=1; u_vec[4]=1
        v_vec = [0]*15; v_vec[0*5+k_offset]=1
        w_vec = [0]*10
        # C10 (k=k_offset, i=1) -> idx k_offset*2+1
        w_vec[k_offset*2+1] += 1
        
        for r in range(6):
            if U_col[r] != 0: U[r][col] = U_col[r] # BUG: U_col is from M1
        # Fix: Use local vars
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

        # M3: a(f-h)
        u_vec = [0]*6; u_vec[0]=1
        v_vec = [0]*15; v_vec[0*5+(k_offset+1)]=1; v_vec[1*5+(k_offset+1)]=-1
        w_vec = [0]*10
        # C01 (k=k_offset+1, i=0) -> idx (k_offset+1)*2+0
        w_vec[(k_offset+1)*2+0] += 1
        # C11 (k=k_offset+1, i=1) -> idx (k_offset+1)*2+1
        w_vec[(k_offset+1)*2+1] += 1
        
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

        # M4: d(g-e)
        u_vec = [0]*6; u_vec[4]=1
        v_vec = [0]*15; v_vec[1*5+k_offset]=1; v_vec[0*5+k_offset]=-1
        w_vec = [0]*10
        # C00 (k=k_offset, i=0) -> idx k_offset*2+0
        w_vec[k_offset*2+0] += 1
        # C10 (k=k_offset, i=1) -> idx k_offset*2+1
        w_vec[k_offset*2+1] += 1
        
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

        # M5: (a+b)h
        u_vec = [0]*6; u_vec[0]=1; u_vec[1]=1
        v_vec = [0]*15; v_vec[1*5+(k_offset+1)]=1
        w_vec = [0]*10
        # C00 (k=k_offset, i=0) -> idx k_offset*2+0
        w_vec[k_offset*2+0] -= 1
        # C01 (k=k_offset+1, i=0) -> idx (k_offset+1)*2+0
        w_vec[(k_offset+1)*2+0] += 1
        
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

        # M6: (c-a)(e+f)
        u_vec = [0]*6; u_vec[3]=1; u_vec[0]=-1
        v_vec = [0]*15; v_vec[0*5+k_offset]=1; v_vec[0*5+(k_offset+1)]=1
        w_vec = [0]*10
        # C11 (k=k_offset+1, i=1) -> idx (k_offset+1)*2+1
        w_vec[(k_offset+1)*2+1] += 1
        
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

        # M7: (b-d)(g+h)
        u_vec = [0]*6; u_vec[1]=1; u_vec[4]=-1
        v_vec = [0]*15; v_vec[1*5+k_offset]=1; v_vec[1*5+(k_offset+1)]=1
        w_vec = [0]*10
        # C00 (k=k_offset, i=0) -> idx k_offset*2+0
        w_vec[k_offset*2+0] += 1
        
        for r in range(6):
            if u_vec[r] != 0: U[r][col] = u_vec[r]
        for r in range(15):
            if v_vec[r] != 0: V[r][col] = v_vec[r]
        for r in range(10):
            if w_vec[r] != 0: W[r][col] = w_vec[r]
        col += 1

    # Call for B_T1 (k_offset=0)
    add_strassen_block(0)
    
    # 3. A_L B_T2a (Rank 7)
    # k_offset=2
    add_strassen_block(2)
    
    # 4. A_L B_T2b (Rank 4)
    # k=4
    # a*x
    add_term(0*3+0, 0*5+4, 4*2+0, 1)
    # b*y
    add_term(0*3+1, 1*5+4, 4*2+0, 1)
    # c*x
    add_term(1*3+0, 0*5+4, 4*2+1, 1)
    # d*y
    add_term(1*3+1, 1*5+4, 4*2+1, 1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
