"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen M_2,2,2 for A_L (cols 0,1) and B_1A (rows 0,1, cols 0,1)
    # A_L: A00(U0), A01(U1), A10(U3), A11(U4)
    # B_1A: B00(V0), B01(V1), B10(V4), B11(V5)
    # C: C00(W0), C01(W2), C10(W1), C11(W3) -- Note: W[k*n+i]
    # C00: k=0,i=0 -> 0
    # C01: k=1,i=0 -> 2
    # C10: k=0,i=1 -> 1
    # C11: k=1,i=1 -> 3
    
    # M1 = (A00+A11)(B00+B11)
    add_term(0, 0, 0, 1, 1, 1) # A00 B00 C00
    add_term(0, 5, 0, 1, 1, 1) # A00 B11 C00
    add_term(4, 0, 0, 1, 1, 1) # A11 B00 C00
    add_term(4, 5, 0, 1, 1, 1) # A11 B11 C00
    
    # Wait, Strassen is 7 terms. I need to implement the 7 multiplications.
    # The add_term function above adds 1 rank-1 tensor.
    # Strassen M1 is a product of sums. It contributes to multiple C entries.
    # It is not a single rank-1 tensor in the final decomposition.
    # The rank-1 tensors are the intermediate products M1..M7.
    # Each Mr is a product of a linear form in A and a linear form in B.
    # So each Mr corresponds to ONE column in U, V, W.
    
    # Let's restart the Strassen implementation.
    
    # Reset U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_strassen_block(a_indices, b_indices, c_indices):
        """
        a_indices: [a00, a01, a10, a11] in U
        b_indices: [b00, b01, b10, b11] in V
        c_indices: [c00, c01, c10, c11] in W
        """
        nonlocal col
        
        # M1 = (a00+a11)(b00+b11)
        U[a_indices[0]][col] += 1
        U[a_indices[3]][col] += 1
        V[b_indices[0]][col] += 1
        V[b_indices[3]][col] += 1
        W[c_indices[0]][col] += 1  # C00
        W[c_indices[3]][col] += 1  # C11
        col += 1
        
        # M2 = (a10+a11)b00
        U[a_indices[2]][col] += 1
        U[a_indices[3]][col] += 1
        V[b_indices[0]][col] += 1
        W[c_indices[1]][col] += 1  # C10
        col += 1
        
        # M3 = a00(b01-b11)
        U[a_indices[0]][col] += 1
        V[b_indices[1]][col] += 1
        V[b_indices[3]][col] += -1
        W[c_indices[2]][col] += 1  # C01
        col += 1
        
        # M4 = a11(b10-b00)
        U[a_indices[3]][col] += 1
        V[b_indices[2]][col] += 1
        V[b_indices[0]][col] += -1
        W[c_indices[1]][col] += 1  # C10
        W[c_indices[0]][col] += 1  # C00
        col += 1
        
        # M5 = (a00+a01)b11
        U[a_indices[0]][col] += 1
        U[a_indices[1]][col] += 1
        V[b_indices[3]][col] += 1
        W[c_indices[0]][col] += -1 # C00
        W[c_indices[2]][col] += 1  # C01
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[a_indices[2]][col] += 1
        U[a_indices[0]][col] += -1
        V[b_indices[0]][col] += 1
        V[b_indices[1]][col] += 1
        W[c_indices[3]][col] += 1  # C11
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[a_indices[1]][col] += 1
        U[a_indices[3]][col] += -1
        V[b_indices[2]][col] += 1
        V[b_indices[3]][col] += 1
        W[c_indices[0]][col] += 1  # C00
        col += 1

    # Block 1: A_L (U0, U1, U3, U4) and B_1A (V0, V1, V4, V5) -> C (W0, W2, W1, W3)
    add_strassen_block([0, 1, 3, 4], [0, 1, 4, 5], [0, 2, 1, 3])
    
    # Block 2: A_L (U0, U1, U3, U4) and B_1B (V2, V3, V6, V7) -> C (W4, W6, W5, W7)
    add_strassen_block([0, 1, 3, 4], [2, 3, 6, 7], [4, 6, 5, 7])
    
    # Schoolbook for A_R (U2, U5) and B_3 (V8, V9, V10, V11)
    # C00(W0), C10(W1), C01(W2), C11(W3), C02(W4), C12(W5), C03(W6), C13(W7)
    
    # C00 = A02 B20
    U[2][col] += 1; V[8][col] += 1; W[0][col] += 1; col += 1
    # C10 = A12 B20
    U[5][col] += 1; V[8][col] += 1; W[1][col] += 1; col += 1
    # C01 = A02 B21
    U[2][col] += 1; V[9][col] += 1; W[2][col] += 1; col += 1
    # C11 = A12 B21
    U[5][col] += 1; V[9][col] += 1; W[3][col] += 1; col += 1
    # C02 = A02 B22
    U[2][col] += 1; V[10][col] += 1; W[4][col] += 1; col += 1
    # C12 = A12 B22
    U[5][col] += 1; V[10][col] += 1; W[5][col] += 1; col += 1
    # C03 = A02 B23
    U[2][col] += 1; V[11][col] += 1; W[6][col] += 1; col += 1
    # C13 = A12 B23
    U[5][col] += 1; V[11][col] += 1; W[7][col] += 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
