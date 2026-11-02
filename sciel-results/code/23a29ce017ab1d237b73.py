import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44 decomposition
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)] # 6 x 44
    V = [[0] * rank for _ in range(m * p)] # 24 x 44
    W = [[0] * rank for _ in range(p * n)] # 16 x 44
    
    col = 0
    
    # Part 1: A1 (2x1) * B1 (1x8)
    # A1 is column 0 of A. Indices in U: i*3 + 0 -> 0, 3
    # B1 is row 0 of B. Indices in V: 0*8 + k -> 0..7
    # C part is 2x8. Indices in W: k*2 + i
    for i in range(n): # 0, 1
        for k in range(p): # 0..7
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Part 2: A2 (2x2) * B2 (2x8)
    # A2 is columns 1,2 of A. Indices in U:
    # a = A[0,1] -> idx 1
    # b = A[0,2] -> idx 2
    # c = A[1,1] -> idx 4
    # d = A[1,2] -> idx 5
    
    # B2 is rows 1,2 of B. Indices in V:
    # e = B[1, 2g]   -> idx 1*8 + 2g
    # f = B[1, 2g+1] -> idx 1*8 + 2g + 1
    # h = B[2, 2g]   -> idx 2*8 + 2g
    # i = B[2, 2g+1] -> idx 2*8 + 2g + 1
    
    # C part is 2x8. Indices in W for group g (cols 2g, 2g+1):
    # C[0, 2g]   -> idx 2g*2 + 0 = 4g
    # C[0, 2g+1] -> idx (2g+1)*2 + 0 = 4g+2
    # C[1, 2g]   -> idx 2g*2 + 1 = 4g+1
    # C[1, 2g+1] -> idx (2g+1)*2 + 1 = 4g+3
    
    # Strassen coefficients for M1..M7
    # Each is a tuple (coeff_a, coeff_b, coeff_c, coeff_d, coeff_e, coeff_f, coeff_h, coeff_i, w_00, w_01, w_10, w_11)
    # w_00 corresponds to C[0,0] of the block, etc.
    
    strassen_terms = [
        # M1 = (a+d)(e+i)
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        # M2 = (c+d)e
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0),
        # M3 = a(f-i)
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 0),
        # M4 = d(h-f)
        (0, 0, 0, 1, 0, -1, 1, 0, -1, 0, 1, 0),
        # M5 = (a+b)i
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        # M6 = (c-a)(e+h)
        (-1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1),
        # M7 = (b-d)(f+i)
        (0, 1, 0, -1, 0, 1, 0, 1, 1, 0, 0, -1),
    ]
    
    for g in range(4): # 4 groups of 2 columns
        base_v_e = 1 * p + 2 * g
        base_v_f = 1 * p + 2 * g + 1
        base_v_h = 2 * p + 2 * g
        base_v_i = 2 * p + 2 * g + 1
        
        base_w_00 = 2 * g * n + 0
        base_w_01 = (2 * g + 1) * n + 0
        base_w_10 = 2 * g * n + 1
        base_w_11 = (2 * g + 1) * n + 1
        
        for term in strassen_terms:
            ca, cb, cc, cd, ce, cf, ch, ci, w00, w01, w10, w11 = term
            
            # Fill U
            if ca != 0: U[0 * m + 1][col] += ca
            if cb != 0: U[0 * m + 2][col] += cb
            if cc != 0: U[1 * m + 1][col] += cc
            if cd != 0: U[1 * m + 2][col] += cd
            
            # Fill V
            if ce != 0: V[base_v_e][col] += ce
            if cf != 0: V[base_v_f][col] += cf
            if ch != 0: V[base_v_h][col] += ch
            if ci != 0: V[base_v_i][col] += ci
            
            # Fill W
            if w00 != 0: W[base_w_00][col] += w00
            if w01 != 0: W[base_w_01][col] += w01
            if w10 != 0: W[base_w_10][col] += w10
            if w11 != 0: W[base_w_11][col] += w11
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
