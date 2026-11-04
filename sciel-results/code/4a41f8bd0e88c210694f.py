import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 132
    rank = 132
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 coefficients
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Schoolbook 2x2x1 coefficients
    # P1 = a*u
    # P2 = b*u
    # P3 = c*v
    # P4 = d*v
    
    # Output mapping for Strassen
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Output mapping for Schoolbook
    # C02 = P1 + P2
    # C12 = P3 + P4
    
    strassen_defs = [
        # (U_coeffs, V_coeffs, W_coeffs)
        # U is 2x2, V is 2x2, W is 2x2
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}), # Wait, M2=(c+d)e. C10+=M2, C11-=M2.
        # Let's check C11 = M1 - M2 + ...
        # So W for M2: C10 gets +1, C11 gets -1.
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}), # M3=a(f-h). C01+=M3, C11+=M3.
        # M4
        ({(1,1):1}, {(1,0):1, (1,1):-1}, {(0,0):1, (1,0):1}), # M4=d(g-c). C00+=M4, C10+=M4.
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}), # M5=(a+b)h. C00-=M5, C01+=M5.
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6=(c-a)(e+f). C11+=M6.
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7=(b-d)(g+h). C00+=M7.
    ]
    
    schoolbook_defs = [
        # P1 = a*u
        ({(0,0):1}, {(0,0):1}, {(0,0):1}),
        # P2 = b*u
        ({(0,1):1}, {(0,0):1}, {(0,0):1}),
        # P3 = c*v
        ({(1,0):1}, {(1,0):1}, {(1,0):1}),
        # P4 = d*v
        ({(1,1):1}, {(1,0):1}, {(1,0):1}),
    ]
    
    # Note: W indices for schoolbook are (0,0) for C02?
    # Local k' for 3rd col is 2.
    # So W indices should be (0,2) and (1,2).
    
    # Correcting Schoolbook W indices
    schoolbook_defs = [
        ({(0,0):1}, {(0,0):1}, {(0,2):1}), # P1 -> C02
        ({(0,1):1}, {(0,0):1}, {(0,2):1}), # P2 -> C02
        ({(1,0):1}, {(1,0):1}, {(1,2):1}), # P3 -> C12
        ({(1,1):1}, {(1,0):1}, {(1,2):1}), # P4 -> C12
    ]
    
    for l in range(3):
        for r in range(2):
            for c in range(2):
                # Process 7 Strassen mults
                for u_def, v_def, w_def in strassen_defs:
                    for (li, lj), val in u_def.items():
                        gi = 2*r + li
                        gj = 2*c + lj
                        U[gi*4 + gj][col] += val
                    for (lj, lk), val in v_def.items():
                        gj = 2*c + lj
                        gk = 3*l + lk
                        V[gj*9 + gk][col] += val
                    for (lk, li), val in w_def.items():
                        gk = 3*l + lk
                        gi = 2*r + li
                        W[gk*4 + gi][col] += val
                    col += 1
                
                # Process 4 Schoolbook mults
                for u_def, v_def, w_def in schoolbook_defs:
                    for (li, lj), val in u_def.items():
                        gi = 2*r + li
                        gj = 2*c + lj
                        U[gi*4 + gj][col] += val
                    for (lj, lk), val in v_def.items():
                        gj = 2*c + lj
                        gk = 3*l + lk
                        V[gj*9 + gk][col] += val
                    for (lk, li), val in w_def.items():
                        gk = 3*l + lk
                        gi = 2*r + li
                        W[gk*4 + gi][col] += val
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
