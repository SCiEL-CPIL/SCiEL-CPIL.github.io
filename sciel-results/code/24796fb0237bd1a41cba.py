"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose the 2x6 matrix into three 2x2 blocks.
    # Each 2x2 block multiplication has rank 7 (Strassen).
    # Total rank = 3 * 7 = 21.
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Coefficients for U (A part), V (B part), W (C part)
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # For a specific block b (0, 1, 2), the k indices are 2b, 2b+1.
    # The j indices are 0, 1.
    # The i indices are 0, 1.
    
    # Define the 7 terms
    # term: (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict {(i,j): val}
    # v_coeffs: dict {(j,k_local): val}
    # w_coeffs: dict {(i,k_local): val}
    
    # Note: k in V and W is global index.
    # For block b, k = 2b + k_local.
    
    terms = [
        # M1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}), # Wait, W is sum for C.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's define the contribution of each M_r to C_{i, k_local}
        # M1: C00 += 1, C11 += 1
        # M2: C10 += 1, C11 -= 1
        # M3: C01 += 1, C11 += 1
        # M4: C00 += 1, C10 += 1
        # M5: C00 -= 1, C01 += 1
        # M6: C11 += 1
        # M7: C00 += 1
    ]
    
    # Correct W coefficients (contribution to C)
    w_terms = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): -1},      # M2
        {(0,1): 1, (1,1): 1},       # M3
        {(0,0): 1, (1,0): 1},       # M4
        {(0,0): -1, (0,1): 1},      # M5
        {(1,1): 1},                 # M6
        {(0,0): 1}                  # M7
    ]
    
    u_terms = [
        {(0,0): 1, (1,1): 1},       # M1: A00+A11
        {(1,0): 1, (1,1): 1},       # M2: A10+A11
        {(0,0): 1},                 # M3: A00
        {(1,1): 1},                 # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00+A01
        {(1,0): 1, (0,0): -1},      # M6: A10-A00
        {(0,1): 1, (1,1): -1}       # M7: A01-A11
    ]
    
    v_terms = [
        {(0,0): 1, (1,1): 1},       # M1: B00+B11
        {(0,0): 1},                 # M2: B00
        {(0,1): 1, (1,1): -1},      # M3: B01-B11
        {(1,0): 1, (0,0): -1},      # M4: B10-B00
        {(1,1): 1},                 # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00+B01
        {(1,0): 1, (1,1): 1}        # M7: B10+B11
    ]
    
    col = 0
    for b in range(3):
        for r in range(7):
            # Fill U
            for (i, j), val in u_terms[r].items():
                U[i * m + j][col] += val
            # Fill V
            for (j, k_local), val in v_terms[r].items():
                k_global = 2 * b + k_local
                V[j * p + k_global][col] += val
            # Fill W
            for (i, k_local), val in w_terms[r].items():
                k_global = 2 * b + k_local
                W[k_global * n + i][col] += val
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
