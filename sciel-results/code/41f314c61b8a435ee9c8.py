"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """Construct a rank-42 decomposition using Strassen's algorithm on 2x2 blocks."""
    n, m, p = DIMENSIONS
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B10)
    # M7 = (A01 - A11)(B01 + B11)
    
    # u_coeffs: list of 7 dicts {(i,j): coeff}
    # v_coeffs: list of 7 dicts {(j,k): coeff}
    # w_coeffs: list of 7 dicts {(k,i): coeff}
    
    u_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1}, # Wait, M5 is (A00 + A10)B11? No, (A00+A10) is row 0?
                             # Standard Strassen:
                             # M5 = (A00 + A10) B11 ? No.
                             # Let's check standard Strassen.
                             # M1 = (A00+A11)(B00+B11)
                             # M2 = (A10+A11)B00
                             # M3 = A00(B10-B11)
                             # M4 = A11(B01-B00)
                             # M5 = (A00+A10)B11
                             # M6 = (A11-A00)(B00+B10)
                             # M7 = (A01-A11)(B01+B11)
                             # C00 = M1 + M4 - M5 + M7
                             # C01 = M3 + M5
                             # C10 = M2 + M4
                             # C11 = M1 - M2 + M3 + M6
                             
                             # Check M5: (A00 + A10) B11.
                             # A00 is (0,0), A10 is (1,0).
                             # So u5 has (0,0):1, (1,0):1.
                             # v5 has (1,1):1.
    ]
    
    # Correcting u_coeffs based on standard Strassen:
    u_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(1,0): 1, (1,1): 1},   # M2
        {(0,0): 1},             # M3
        {(1,1): 1},             # M4
        {(0,0): 1, (1,0): 1},   # M5
        {(1,1): 1, (0,0): -1},  # M6
        {(0,1): 1, (1,1): -1}   # M7
    ]
    
    v_coeffs = [
        {(0,0): 1, (1,1): 1},   # M1
        {(0,0): 1},             # M2
        {(1,0): 1, (1,1): -1},  # M3
        {(0,1): 1, (0,0): -1},  # M4
        {(1,1): 1},             # M5
        {(0,0): 1, (1,0): 1},   # M6
        {(0,1): 1, (1,1): 1}    # M7
    ]
    
    w_coeffs = [
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}, # M1: C00+=M1, C11+=M1
        {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}, # M2: C10+=M2, C11-=M2
        {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1},  # M3: C01+=M3, C11+=M3
        {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0},  # M4: C00+=M4, C10+=M4
        {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}, # M5: C00-=M5, C01+=M5
        {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1},  # M6: C11+=M6
        {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}   # M7: C00+=M7
    ]
    
    # Verify w_coeffs:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # w1: C00:1, C11:1. Correct.
    # w2: C10:1, C11:-1. Correct.
    # w3: C01:1, C11:1. Correct.
    # w4: C00:1, C10:1. Correct.
    # w5: C00:-1, C01:1. Correct.
    # w6: C11:1. Correct.
    # w7: C00:1. Correct.
    
    col = 0
    for b in range(6):
        s = b // 3
        q = b % 3
        j_base = 2 * s
        k_base = 2 * q
        
        for m_idx in range(7):
            # Update U
            for (i_loc, j_loc), coeff in u_coeffs[m_idx].items():
                i_glob = i_loc
                j_glob = j_base + j_loc
                u_idx = i_glob * m + j_glob
                U[u_idx][col] += coeff
            
            # Update V
            for (j_loc, k_loc), coeff in v_coeffs[m_idx].items():
                j_glob = j_base + j_loc
                k_glob = k_base + k_loc
                v_idx = j_glob * p + k_glob
                V[v_idx][col] += coeff
            
            # Update W
            for (k_loc, i_loc), coeff in w_coeffs[m_idx].items():
                k_glob = k_base + k_loc
                i_glob = i_loc
                w_idx = k_glob * n + i_glob
                W[w_idx][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
