"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 products definition
    # Each product is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict { (i, j): coeff }
    # V_coeffs: dict { (j, k): coeff }
    # W_coeffs: dict { (k, i): coeff }
    
    strassen_products = [
        # P1 = (A00 + A11)(B00 + B11)
        # C00 += 1, C11 += 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A10 + A11) B00
        # C10 += 1, C11 -= 1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # P3 = A00 (B01 - B11)
        # C01 += 1, C11 += 1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # P4 = A11 (B10 - B00)
        # C00 += 1, C10 += 1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}),
        # P5 = (A00 + A01) B11
        # C00 -= 1, C01 += 1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}),
        # P6 = (A10 - A00) (B00 + B01)
        # C11 += 1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A01 - A11) (B10 + B11)
        # C00 += 1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Iterate over the 8 blocks
    # ar determines the intermediate row block (j, j+1)
    # qc determines the output column block (k, k+1)
    for ar in range(2):
        for qc in range(4):
            # Global indices offsets
            # A columns: 2*ar, 2*ar+1
            # B rows: 2*ar, 2*ar+1
            # B columns: 2*qc, 2*qc+1
            # C columns: 2*qc, 2*qc+1
            # C rows: 0, 1
            
            for u_coeff, v_coeff, w_coeff in strassen_products:
                # Fill U column col
                # U index: i*m + j
                for (li, lj), coeff in u_coeff.items():
                    gi = li
                    gj = 2 * ar + lj
                    U[gi * m + gj][col] += coeff
                
                # Fill V column col
                # V index: j*p + k
                for (lj, lk), coeff in v_coeff.items():
                    gj = 2 * ar + lj
                    gk = 2 * qc + lk
                    V[gj * p + gk][col] += coeff
                
                # Fill W column col
                # W index: k*n + i
                for (lk, li), coeff in w_coeff.items():
                    gk = 2 * qc + lk
                    gi = li
                    W[gk * n + gi][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
