"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition by summing contributions from sub-problems.
    # Sub-problems:
    # 1. j in {0, 1}, k in {0, 1, 2, 3, 4, 5} -> 3 blocks of 2x2x2 (Strassen)
    # 2. j in {0, 1}, k in {6} -> 1 block of 2x2x1 (Trivial)
    # 3. j in {2}, k in {0..6} -> 1 block of 2x1x7 (Rank 1)
    
    # Initialize matrices
    rank = 26 # 3*7 + 4 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to add a term
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal current_rank
        r = current_rank
        current_rank += 1
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i * m + j][r] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j * p + k][r] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k * n + i][r] += val

    # 1. Strassen for 2x2x2 blocks
    # Strassen coefficients
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B10-B11) -> C01, C11
    # M4 = A11(B01-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B10) -> C11
    # M7 = (A01-A11)(B11+B01) -> C00, C01
    
    def add_strassen_block(k_offset):
        # k_offset is the starting column index for this 2x2x2 block
        # Indices: i in 0..1, j in 0..1, k in k_offset..k_offset+1
        
        # M1
        add_term({(0,0):1, (1,1):1}, {(0, k_offset):1, (1, k_offset+1):1}, {(k_offset, 0):1, (k_offset+1, 1):1})
        # M2
        add_term({(1,0):1, (1,1):1}, {(0, k_offset):1}, {(k_offset, 1):1, (k_offset+1, 1):-1})
        # M3
        add_term({(0,0):1}, {(1, k_offset):1, (1, k_offset+1):-1}, {(k_offset, 0):1, (k_offset+1, 1):1})
        # M4
        add_term({(1,1):1}, {(0, k_offset+1):1, (0, k_offset):-1}, {(k_offset, 0):-1, (k_offset, 1):1})
        # M5
        add_term({(0,0):1, (0,1):1}, {(1, k_offset+1):1}, {(k_offset, 0):-1, (k_offset, 0):1}) # Wait, C00 gets -M5, C01 gets +M5
        # Correcting M5:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-define add_term to be clearer or just hardcode the W coefficients.
        # I will rewrite the add_term calls with correct W coefficients.
        
    # Reset current_rank and redo
    current_rank = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(k_offset):
        nonlocal current_rank
        
        # M1
        r = current_rank; current_rank += 1
        U[0*m+0][r] = 1; U[1*m+1][r] = 1
        V[0*p+k_offset][r] = 1; V[1*p+k_offset+1][r] = 1
        W[k_offset*n+0][r] = 1; W[(k_offset+1)*n+1][r] = 1
        
        # M2
        r = current_rank; current_rank += 1
        U[1*m+0][r] = 1; U[1*m+1][r] = 1
        V[0*p+k_offset][r] = 1
        W[k_offset*n+1][r] = 1; W[(k_offset+1)*n+1][r] = -1
        
        # M3
        r = current_rank; current_rank += 1
        U[0*m+0][r] = 1
        V[1*p+k_offset][r] = 1; V[1*p+k_offset+1][r] = -1
        W[k_offset*n+0][r] = 1; W[(k_offset+1)*n+1][r] = 1
        
        # M4
        r = current_rank; current_rank += 1
        U[1*m+1][r] = 1
        V[0*p+k_offset+1][r] = 1; V[0*p+k_offset][r] = -1
        W[k_offset*n+0][r] = -1; W[k_offset*n+1][r] = 1
        
        # M5
        r = current_rank; current_rank += 1
        U[0*m+0][r] = 1; U[0*m+1][r] = 1
        V[1*p+k_offset+1][r] = 1
        W[k_offset*n+0][r] = -1; W[k_offset*n+1][r] = 1
        
        # M6
        r = current_rank; current_rank += 1
        U[1*m+0][r] = 1; U[0*m+0][r] = -1
        V[0*p+k_offset][r] = 1; V[1*p+k_offset][r] = 1
        W[(k_offset+1)*n+1][r] = 1
        
        # M7
        r = current_rank; current_rank += 1
        U[0*m+1][r] = 1; U[1*m+1][r] = -1
        V[1*p+k_offset+1][r] = 1; V[0*p+k_offset+1][r] = 1
        W[k_offset*n+0][r] = 1; W[k_offset*n+1][r] = 1

    # Add 3 Strassen blocks for k=0, 2, 4
    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)
    
    # 2. Trivial for 2x2x1 at k=6
    # C_i6 = sum_j A_ij B_j6
    # Rank 4
    for i in range(2):
        for j in range(2):
            r = current_rank; current_rank += 1
            U[i*m+j][r] = 1
            V[j*p+6][r] = 1
            W[6*n+i][r] = 1
            
    # 3. Rank 1 for 2x1x7 (j=2)
    # C_i k = A_i2 B_2k
    # U_i2 = 1, V_2k = 1, W_ki = 1
    r = current_rank; current_rank += 1
    for i in range(2):
        U[i*m+2][r] = 1
    for k in range(7):
        V[2*p+k][r] = 1
    for k in range(7):
        for i in range(2):
            W[k*n+i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
