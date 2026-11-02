"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen coefficients for 2x2 multiplication
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B local indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C local indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    strassen = [
        # t=0: M1 = (A00+A11)(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {(0, 0): 1, (1, 1): 1}),
        # t=1: M2 = (A10+A11)B00
        ({2: 1, 3: 1}, {0: 1}, {(1, 0): 1, (1, 1): -1}),
        # t=2: M3 = A00(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {(0, 1): 1, (1, 1): 1}),
        # t=3: M4 = A11(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {(0, 0): 1, (1, 0): 1}),
        # t=4: M5 = (A00+A01)B11
        ({0: 1, 1: 1}, {3: 1}, {(0, 0): -1, (0, 1): 1}),
        # t=5: M6 = (A10-A00)(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {(1, 1): 1}),
        # t=6: M7 = (A01-A11)(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {(0, 0): 1})
    ]
    
    r = 0
    for b in range(3):  # 3 blocks of 2 columns each
        for t in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen[t]
            
            # Fill U (same for all blocks)
            for idx, val in u_coeffs.items():
                U[idx][r] = val
                
            # Fill V: map local B index to global V index
            # Local l: 0->(j=0,k=2b), 1->(j=0,k=2b+1), 2->(j=1,k=2b), 3->(j=1,k=2b+1)
            # Global V index = j*p + k = j*6 + k
            for l, val in v_coeffs.items():
                if l == 0:
                    v_idx = 0 * p + (2 * b)      # j=0, k=2b
                elif l == 1:
                    v_idx = 0 * p + (2 * b + 1)  # j=0, k=2b+1
                elif l == 2:
                    v_idx = 1 * p + (2 * b)      # j=1, k=2b
                elif l == 3:
                    v_idx = 1 * p + (2 * b + 1)  # j=1, k=2b+1
                else:
                    continue
                V[v_idx][r] = val
                
            # Fill W: W index = k*n + i
            # w_coeffs keys are (i, k_local) where k_local is 0 or 1
            # k_global = 2*b + k_local
            for (i, k_local), val in w_coeffs.items():
                k_global = 2 * b + k_local
                w_idx = k_global * n + i
                W[w_idx][r] = val
                
            r += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
