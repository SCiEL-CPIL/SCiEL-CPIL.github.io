"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # rank 51 decomposition
    rank = 51
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c_idx = 0
    
    # Part 1: A1 * B1 (18 terms)
    # A1 is col 0 of A: indices 0, 3
    # B1 is row 0 of B: indices 0..8
    for i in range(n):
        a_idx = i * m  # 0 or 3
        for k in range(p):
            b_idx = k  # 0..8
            w_idx = k * n + i
            
            U[a_idx][c_idx] = 1
            V[b_idx][c_idx] = 1
            W[w_idx][c_idx] = 1
            c_idx += 1
            
    # Part 2: A2 * B2 (33 terms)
    # A2 is cols 1,2 of A: 
    # a = A[0,1] (idx 1), b = A[0,2] (idx 2)
    # c = A[1,1] (idx 4), d = A[1,2] (idx 5)
    # B2 is rows 1,2 of B.
    # Split B2 into 3 blocks of 3 cols.
    for block in range(3):
        # B block indices
        # Row 1 of B: 9 + 3*block ... 9 + 3*block + 2
        # Row 2 of B: 18 + 3*block ... 18 + 3*block + 2
        e_idx = 9 + 3 * block
        f_idx = e_idx + 1
        g_idx = e_idx + 2
        h_idx = 18 + 3 * block
        i_idx = h_idx + 1
        j_idx = h_idx + 2
        
        a_idx = 1
        b_idx = 2
        c_row_idx = 4
        d_idx = 5
        
        # Strassen 2x2x2 for first 2 cols of block
        # P1 = (a+b)(e+h)
        U[a_idx][c_idx] = 1; U[b_idx][c_idx] = 1
        V[e_idx][c_idx] = 1; V[h_idx][c_idx] = 1
        # W contributions for P1
        # C00 += a e + b h => P1 contributes a e + b h + a h + b e
        # We need to extract a e and b h from P1 using other P's
        # Standard Strassen reconstruction:
        # C00 = P5 + P4 - P6 + P2  (using indices from a specific formulation)
        # Let's use the explicit formulas:
        # P1 = (a+b)(e+h) = ae + ah + be + bh
        # P2 = (c+d)(e+h) = ce + ch + de + dh
        # P3 = (a-c)(e+h) = ae + ah - ce - ch
        # P4 = (a+b)(i-h) = ai - ah + bi - bh
        # P5 = a e
        # P6 = b i
        # P7 = c h
        # C00 = P5 + P7 - P6 + P4? No.
        # C00 = ae + bh
        # P5 = ae
        # P7 = ch
        # P6 = bi
        # This doesn't give bh directly.
        
        # Let's use the 7 products:
        # M1 = (a+b)(e+f) -- wait, B is 2x2 for Strassen.
        # B' = [[e, f], [h, i]]
        # M1 = (a+b)(e+h)
        # M2 = (c+d)(e+h)
        # M3 = (a-c)(e+h)
        # M4 = (a+b)(i-h)
        # M5 = a e
        # M6 = b i
        # M7 = c h
        # C00 = M5 + M4 - M6 + M7?
        # M4 = ai - ah + bi - bh
        # M5 = ae
        # M6 = bi
        # M7 = ch
        # M5 + M4 - M6 + M7 = ae + ai - ah + bi - bh - bi + ch
        # = ae + ai - ah - bh + ch. Not C00.
        
        # Correct Strassen reconstruction:
        # C00 = M5 + M2 - M4 - M7 + M6?
        # Let's stick to the 12 schoolbook terms if I can't get Strassen right quickly.
        # But 12*3 + 18 = 54.
        
        # I will use 12 schoolbook terms for A2*B2 to be safe.
        # Total rank 54.
        # This is not an improvement.
        
        # Let's try to implement the 11 terms correctly.
        # I will use the known 11-term decomposition for 2x2x3.
        
        # Term 1: (a+b)(e+h)
        U[a_idx][c_idx] = 1; U[b_idx][c_idx] = 1
        V[e_idx][c_idx] = 1; V[h_idx][c_idx] = 1
        W[0][c_idx] = 0 # Will be combined later? No, we need to add to W.
        # Since we are constructing the tensor directly, we must add the linear combination.
        # C00 gets +1 * P1
        # C01 gets +1 * P1? No.
        
        # This is getting complicated to verify without running.
        # I will return the schoolbook 54 to ensure validity, as 51 is risky without verification.
        pass
        
    # Fallback to Schoolbook 54 if optimization fails
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
