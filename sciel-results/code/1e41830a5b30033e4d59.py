"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    """Strassen-based rank-56 decomposition for 2x2 times 2x16 matrix multiplication."""
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_groups = p // 2  # 8 groups of 2 columns
    rank = 7 * num_groups  # 56
    
    U = [[0] * rank for _ in range(n * m)]  # shape (4, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (32, 56)
    
    for j in range(num_groups):
        base = j * 7
        # V indices for B entries in group j:
        # B[0, 2j]   -> V[16*j]
        # B[0, 2j+1] -> V[16*j + 1]
        # B[1, 2j]   -> V[16 + 16*j]
        # B[1, 2j+1] -> V[16 + 16*j + 1]
        v_b00 = 16 * j          # B[0, 2j]
        v_b01 = 16 * j + 1      # B[0, 2j+1]
        v_b10 = 16 + 16 * j     # B[1, 2j]
        v_b11 = 16 + 16 * j + 1 # B[1, 2j+1]
        
        # W indices for C entries in group j:
        # C[0, 2j]   -> W[4*j]
        # C[0, 2j+1] -> W[4*j + 2]
        # C[1, 2j]   -> W[4*j + 1]
        # C[1, 2j+1] -> W[4*j + 3]
        w_c00 = 4 * j
        w_c01 = 4 * j + 2
        w_c10 = 4 * j + 1
        w_c11 = 4 * j + 3
        
        # U indices for A entries:
        # A[0,0] -> U[0], A[0,1] -> U[1], A[1,0] -> U[2], A[1,1] -> U[3]
        
        # Step 0: M1 = (A00+A11)(B00+B11)
        r = base + 0
        U[0][r] = 1; U[3][r] = 1
        V[v_b00][r] = 1; V[v_b11][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        
        # Step 1: M2 = (A10+A11)B00
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_b00][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        
        # Step 2: M3 = A00(B01-B11)
        r = base + 2
        U[0][r] = 1
        V[v_b01][r] = 1; V[v_b11][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        
        # Step 3: M4 = A11(B10-B00)
        r = base + 3
        U[3][r] = 1
        V[v_b10][r] = 1; V[v_b00][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        
        # Step 4: M5 = (A00+A01)B11
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_b11][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        
        # Step 5: M6 = (A10-A00)(B00+B01)
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_b00][r] = 1; V[v_b01][r] = 1
        W[w_c11][r] = 1
        
        # Step 6: M7 = (A01-A11)(B10+B11)
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_b10][r] = 1; V[v_b11][r] = 1
        W[w_c00][r] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
