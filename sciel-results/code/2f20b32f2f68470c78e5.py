"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 53 decomposition
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a single multiplication term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Helper to add Strassen block
    def add_strassen(a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1):
        # A is 2x2: (a_r0, a_c0), (a_r0, a_c1), (a_r1, a_c0), (a_r1, a_c1)
        # B is 2x2: (b_r0, b_c0), (b_r0, b_c1), (b_r1, b_c0), (b_r1, b_c1)
        # C is 2x2: (b_c0, a_r0), (b_c1, a_r0), (b_c0, a_r1), (b_c1, a_r1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        u00 = a_r0 * m + a_c0
        u11 = a_r1 * m + a_c1
        v00 = b_r0 * p + b_c0
        v11 = b_r1 * p + b_c1
        w00 = b_c0 * n + a_r0
        w11 = b_c1 * n + a_r1
        
        # We need to sum coefficients. 
        # M1 contributes +1 to C00 and +1 to C11.
        # But U and V are linear combinations.
        # U term: A00 + A11. So U[u00]=1, U[u11]=1.
        # V term: B00 + B11. So V[v00]=1, V[v11]=1.
        # W term: C00 += M1, C11 += M1. So W[w00]=1, W[w11]=1.
        
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00
        u10 = a_r1 * m + a_c0
        # U: A10, A11. V: B00. W: C10, C11?
        # C10 = M2 + M4. C11 = M1 - M2 + ...
        # M2 contributes +1 to C10, -1 to C11.
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[b_c0 * n + a_r1][col] += 1 # C10
        W[b_c1 * n + a_r1][col] += -1 # C11
        col += 1
        
        # M3 = A00 * (B01 - B11)
        # U: A00. V: B01, B11. W: C01, C11?
        # C01 = M3 + M5. C11 = ... + M3.
        # M3 contributes +1 to C01, +1 to C11.
        v01 = b_r0 * p + b_c1
        # W C01 index: b_c1 * n + a_r0
        # W C11 index: b_c1 * n + a_r1
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] += -1
        W[b_c1 * n + a_r0][col] += 1 # C01
        W[b_c1 * n + a_r1][col] += 1 # C11
        col += 1
        
        # M4 = A11 * (B10 - B00)
        # U: A11. V: B10, B00. W: C10, C00?
        # C10 = M2 + M4. C00 = M1 + M4 - ...
        # M4 contributes +1 to C10, +1 to C00.
        v10 = b_r1 * p + b_c0
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] += -1
        W[b_c0 * n + a_r1][col] += 1 # C10
        W[b_c0 * n + a_r0][col] += 1 # C00
        col += 1
        
        # M5 = (A00 + A01) * B11
        # U: A00, A01. V: B11. W: C01, C00?
        # C01 = M3 + M5. C00 = M1 + M4 - M5 + ...
        # M5 contributes +1 to C01, -1 to C00.
        u01 = a_r0 * m + a_c1
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[b_c1 * n + a_r0][col] += 1 # C01
        W[b_c0 * n + a_r0][col] += -1 # C00
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10, A00. V: B00, B01. W: C11?
        # C11 = M1 - M2 + M3 + M6.
        # M6 contributes +1 to C11.
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[b_c1 * n + a_r1][col] += 1 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01, A11. V: B10, B11. W: C00?
        # C00 = M1 + M4 - M5 + M7.
        # M7 contributes +1 to C00.
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[b_c0 * n + a_r0][col] += 1 # C00
        col += 1

    # Helper to add Schoolbook 2x2 by 2x1
    def add_schoolbook_221(a_r0, a_c0, a_r1, a_c1, b_r0, b_c0, b_r1, b_c1):
        # A 2x2, B 2x1.
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        # M1 = A00 B00 -> C00
        u00 = a_r0 * m + a_c0
        v00 = b_r0 * p + b_c0
        w00 = b_c0 * n + a_r0
        U[u00][col] += 1
        V[v00][col] += 1
        W[w00][col] += 1
        col += 1
        
        # M2 = A01 B10 -> C00
        u01 = a_r0 * m + a_c1
        v10 = b_r1 * p + b_c0
        U[u01][col] += 1
        V[v10][col] += 1
        W[w00][col] += 1
        col += 1
        
        # M3 = A10 B00 -> C10
        u10 = a_r1 * m + a_c0
        w10 = b_c0 * n + a_r1
        U[u10][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M4 = A11 B10 -> C10
        u11 = a_r1 * m + a_c1
        U[u11][col] += 1
        V[v10][col] += 1
        W[w10][col] += 1
        col += 1

    # Block 1: A[0:2, 0:1] * B[0:1, 0:7] (M_2,1,7)
    # A cols 0, B rows 0
    for i in range(2):
        for k in range(7):
            add_term(i * m + 0, 0 * p + k, k * n + i)
            
    # Block 2: A[0:2, 1:2] * B[1:2, 0:7] (M_2,1,7)
    # A cols 1, B rows 1
    for i in range(2):
        for k in range(7):
            add_term(i * m + 1, 1 * p + k, k * n + i)

    # Block 3: A[0:2, 2:4] * B[2:4, 0:7] (M_2,2,7)
    # A cols 2,3. B rows 2,3.
    # Split B cols into 0,1,2 and 3,4,5,6
    
    # 3a: M_2,2,3 (B cols 0,1,2)
    # 3a-i: M_2,2,2 (B cols 0,1) Strassen
    add_strassen(0, 2, 1, 3, 2, 0, 3, 1)
    
    # 3a-ii: M_2,2,1 (B col 2) Schoolbook
    add_schoolbook_221(0, 2, 1, 3, 2, 2, 3, 2)
    
    # 3b: M_2,2,4 (B cols 3,4,5,6)
    # 3b-i: M_2,2,2 (B cols 3,4) Strassen
    add_strassen(0, 2, 1, 3, 2, 3, 3, 4)
    
    # 3b-ii: M_2,2,2 (B cols 5,6) Strassen
    add_strassen(0, 2, 1, 3, 2, 5, 3, 6)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
