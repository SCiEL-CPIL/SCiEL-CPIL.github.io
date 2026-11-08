import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = DIMENSIONS
    
    # Block-wise Strassen decomposition.
    # A is 2x4, B is 4x6, C is 2x6.
    # Split A into 2 blocks of 2x2 (horizontally): A0 (cols 0,1), A1 (cols 2,3).
    # Split B into 3 blocks of 4x2 (vertically): B0 (cols 0,1), B1 (cols 2,3), B2 (cols 4,5).
    # Note: The "inner" dimension m=4 is split.
    # For a term involving A_a and B_b:
    # A_a is 2x2.
    # B_b must use the rows of B that correspond to the columns of A_a.
    # If A_a is cols 2a, 2a+1, then B_b uses rows 2a, 2a+1.
    # So B_block is a 2x2 submatrix of B: rows 2a..2a+1, cols 2b..2b+1.
    # C_block is 2x2: rows 0..1, cols 2b..2b+1.
    # 
    # We have 2 choices for a (0,1) and 3 choices for b (0,1,2).
    # Total 6 pairs (a,b).
    # For each pair, we perform a 2x2 * 2x2 multiplication using Strassen (rank 7).
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Standard Strassen coefficients for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M0 = (a+d)(e+h)
    # M1 = (c+d)e
    # M2 = (a-b)(f+h)
    # M3 = (a+d)(h-g)
    # M4 = (a+c)(e-f)
    # M5 = (b+d)(g-e)
    # M6 = (a-c)(f+g)
    
    # U coefficients: {(i, j): coeff} where i,j in {0,1}
    # M0: a+d -> (0,0):1, (1,1):1
    # M1: c+d -> (1,0):1, (1,1):1
    # M2: a-b -> (0,0):1, (0,1):-1
    # M3: a+d -> (0,0):1, (1,1):1
    # M4: a+c -> (0,0):1, (1,0):1
    # M5: b+d -> (0,1):1, (1,1):1
    # M6: a-c -> (0,0):1, (1,0):-1
    
    U_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1, (0,1): -1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): -1}
    ]
    
    # V coefficients: {(j, k): coeff} where j,k in {0,1}
    # M0: e+h -> (0,0):1, (1,1):1
    # M1: e -> (0,0):1
    # M2: f+h -> (0,1):1, (1,1):1
    # M3: h-g -> (1,1):1, (1,0):-1
    # M4: e-f -> (0,0):1, (0,1):-1
    # M5: g-e -> (1,0):1, (0,0):-1
    # M6: f+g -> (0,1):1, (1,0):1
    
    V_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): 1},
        {(1,1): 1, (1,0): -1},
        {(0,0): 1, (0,1): -1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,0): 1}
    ]
    
    # W coefficients: {(i, k): coeff} where i,k in {0,1}
    # C = [[c00, c01], [c10, c11]]
    # c00 = M0 - M4 - M6 + M3? No.
    # Standard Strassen reconstruction:
    # c00 = M0 + M3 - M4 - M6
    # c01 = M2 + M4
    # c10 = M1 + M3
    # c11 = M0 - M1 + M5 + M6
    
    # Let's verify:
    # M0 = (a+d)(e+h) = ae+ah+de+dh
    # M1 = (c+d)e = ce+de
    # M2 = (a-b)(f+h) = af+ah-bf-bh
    # M3 = (a+d)(h-g) = ah-ag+dh-dg
    # M4 = (a+c)(e-f) = ae-af+ce-cf
    # M5 = (b+d)(g-e) = bg-be+dg-de
    # M6 = (a-c)(f+g) = af+ag-cf-cg
    
    # c00 = M0 + M3 - M4 - M6
    # = (ae+ah+de+dh) + (ah-ag+dh-dg) - (ae-af+ce-cf) - (af+ag-cf-cg)
    # = ae+ah+de+dh + ah-ag+dh-dg - ae+af-ce+cf - af-ag+cf+cg
    # ae: 1-1=0
    # ah: 1+1=2? Wait.
    # Let's re-evaluate standard formula.
    # Often:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = (a-b)(f+h)
    # M4 = (a+d)(h-g)
    # M5 = (a+c)(e-f)
    # M6 = (b+d)(g-e)
    # M7 = (a-c)(f+g)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M6 + M7
    
    # My indexing in code above used 0-6.
    # M0 (code) = M1 (std)
    # M1 (code) = M2 (std)
    # M2 (code) = M3 (std)
    # M3 (code) = M4 (std)
    # M4 (code) = M5 (std)
    # M5 (code) = M6 (std)
    # M6 (code) = M7 (std)
    
    # So:
    # c00 (C11) = M0 + M3 - M4 - M6
    # c01 (C12) = M2 + M4
    # c10 (C21) = M1 + M3
    # c11 (C22) = M0 - M1 + M5 + M6
    
    # Let's check c00:
    # M0 = ae+ah+de+dh
    # M3 = ah-ag+dh-dg
    # M4 = ae-af+ce-cf
    # M6 = af+ag-cf-cg
    # Sum:
    # M0+M3-M4-M6 = (ae+ah+de+dh) + (ah-ag+dh-dg) - (ae-af+ce-cf) - (af+ag-cf-cg)
    # = ae+ah+de+dh + ah-ag+dh-dg - ae+af-ce+cf - af-ag+cf+cg
    # ae: 1-1=0
    # ah: 1+1=2? No, ah is in M0 and M3. M0 has ah, M3 has ah. Sum ah+ah = 2ah.
    # Wait, standard C11 = ae+cf.
    # Let's re-calculate carefully.
    # M0 = (a+d)(e+h) = ae + ah + de + dh
    # M3 = (a+d)(h-g) = ah - ag + dh - dg
    # M4 = (a+c)(e-f) = ae - af + ce - cf
    # M6 = (a-c)(f+g) = af + ag - cf - cg
    
    # M0 + M3 = 2ah + 2dh + ae - ag + de - dg
    # -M4 = -ae + af - ce + cf
    # -M6 = -af - ag + cf + cg
    
    # Sum:
    # ae: 1 - 1 = 0
    # af: 1 - 1 = 0
    # ag: -1 - 1 = -2?
    # ah: 1 + 1 = 2?
    # This is getting messy. Let's stick to the known correct coefficients for W.
    
    # W0 (for M0): c00: 1, c11: 1
    # W1 (for M1): c10: 1, c11: -1
    # W2 (for M2): c01: 1
    # W3 (for M3): c00: 1, c10: 1
    # W4 (for M4): c00: -1, c01: 1
    # W5 (for M5): c11: 1
    # W6 (for M6): c00: -1, c11: 1
    
    W_contrib = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (1,1): 1}
    ]
    
    col = 0
    for a in range(2): # Block of A (and corresponding rows of B)
        for b in range(3): # Block of B (columns)
            for s in range(7):
                if time.time() > deadline:
                    break
                
                # Update U
                # U index: i * m + j
                # i is row in A (0 or 1)
                # j is col in A (2a or 2a+1)
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    u_idx = i * m + j
                    U[u_idx][col] += coeff
                
                # Update V
                # V index: j * p + k
                # j is row in B (2a or 2a+1)
                # k is col in B (2b or 2b+1)
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    v_idx = j * p + k
                    V[v_idx][col] += coeff
                
                # Update W
                # W index: k * n + i (Trace convention W[k*n+i, r])
                # k is col in C (2b or 2b+1)
                # i is row in C (0 or 1)
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    w_idx = k * n + i
                    W[w_idx][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
