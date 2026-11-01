"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 21 decomposition using Strassen's algorithm on three 2x2 blocks
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 multiplications for 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A11 - A01)(B10 + B11)
    
    # Output C:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices in U (n*m = 4):
    # A00 -> 0, A01 -> 1, A10 -> 2, A11 -> 3
    
    # Indices in V (m*p = 12):
    # For block b (0,1,2), local k (0,1):
    # B00 -> 6*b + 0
    # B01 -> 6*b + 1
    # B10 -> 6*b + 6
    # B11 -> 6*b + 7
    
    # Indices in W (p*n = 12):
    # For block b (0,1,2), local k (0,1), local i (0,1):
    # C00 -> (2*b + 0)*2 + 0 = 4*b
    # C01 -> (2*b + 1)*2 + 0 = 4*b + 2
    # C10 -> (2*b + 0)*2 + 1 = 4*b + 1
    # C11 -> (2*b + 1)*2 + 1 = 4*b + 3
    
    # Define the 7 Strassen terms' contributions to U, V, W
    # Each term is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {idx: val}
    # V_coeffs: dict {idx: val} (relative to block start 6*b)
    # W_coeffs: dict {idx: val} (relative to block start 4*b)
    
    strassen_terms = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 7: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {1: 1}), # C10 gets M2. C10 is 4*b+1.
        # M3
        ({0: 1}, {1: 1, 7: -1}, {2: 1, 3: 1}), # C01 gets M3, C11 gets M3. C01 is 4*b+2, C11 is 4*b+3.
        # M4
        ({3: 1}, {6: 1, 0: -1}, {0: 1, 1: 1}), # C00 gets M4, C10 gets M4.
        # M5
        ({0: 1, 1: 1}, {7: 1}, {0: -1, 2: 1}), # C00 gets -M5, C01 gets M5.
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C11 gets M6.
        # M7
        ({3: 1, 1: -1}, {6: 1, 7: 1}, {0: 1, 1: -1}) # C00 gets M7, C10 gets -M7? 
        # Wait, C10 = M2 + M4. No M7.
        # C00 = M1 + M4 - M5 + M7.
        # C11 = M1 - M2 + M3 + M6.
        # Let's re-verify C10 and C11 for M7.
        # M7 = (A11 - A01)(B10 + B11).
        # Does M7 appear in C10 or C11?
        # Standard Strassen:
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 only appears in C00.
        # Let's re-read standard Strassen formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Yes, M7 is only in C00.
        # So W for M7 should be {0: 1}.
    ]
    
    # Correcting M7 W coefficients
    # M7: U({3:1, 1:-1}), V({6:1, 7:1}), W({0:1})
    strassen_terms[6] = ({3: 1, 1: -1}, {6: 1, 7: 1}, {0: 1})

    for b in range(3):
        for s in range(7):
            r = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for idx, val in u_coeffs.items():
                U[idx][r] += val
            
            for idx, val in v_coeffs.items():
                V[6 * b + idx][r] += val
            
            for idx, val in w_coeffs.items():
                W[4 * b + idx][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
