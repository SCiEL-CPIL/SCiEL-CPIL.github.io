import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 112
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = a(e+h)
    # M2 = (a+b)(g+h)
    # M3 = (c+d)(e+f)
    # M4 = d(f-g)
    # M5 = (a+c)e
    # M6 = (b-d)h
    # M7 = (c+b)(f-h)
    
    # Output C
    # c00 = M1 + M4 - M6 + M7
    # c01 = M2 - M4 + M6
    # c10 = M3 - M5 - M7
    # c11 = M1 - M3 + M5
    
    # We have 2x2x4.
    # B = [[e, f, e', f'], [g, h, g', h']]
    # B_L = [[e, f], [g, h]]
    # B_R = [[e', f'], [g', h']]
    
    # M_L1 = a(e+h)
    # M_L2 = (a+b)(g+h)
    # M_L3 = (c+d)(e+f)
    # M_L4 = d(f-g)
    # M_L5 = (a+c)e
    # M_L6 = (b-d)h
    # M_L7 = (c+b)(f-h)
    
    # M_R1 = a(e'+h')
    # M_R2 = (a+b)(g'+h')
    # M_R3 = (c+d)(e'+f')
    # M_R4 = d(f'-g')
    # M_R5 = (a+c)e'
    # M_R6 = (b-d)h'
    # M_R7 = (c+b)(f'-h')
    
    # C_L = [[c00, c01], [c10, c11]]
    # C_R = [[c02, c03], [c12, c13]]
    
    # c02 = M_R1 + M_R4 - M_R6 + M_R7
    # c03 = M_R2 - M_R4 + M_R6
    # c12 = M_R3 - M_R5 - M_R7
    # c13 = M_R1 - M_R3 + M_R5

    for I in range(2):
        for J in range(2):
            for Side in range(2):
                # A block indices
                a_r0 = 2 * I
                a_r1 = 2 * I + 1
                a_c0 = 2 * J
                a_c1 = 2 * J + 1
                
                # B block indices
                b_r0 = 2 * J
                b_r1 = 2 * J + 1
                b_c0 = 4 * Side
                b_c1 = 4 * Side + 1
                b_c2 = 4 * Side + 2
                b_c3 = 4 * Side + 3
                
                # C block indices
                c_r0 = 2 * I
                c_r1 = 2 * I + 1
                c_c0 = 4 * Side
                c_c1 = 4 * Side + 1
                c_c2 = 4 * Side + 2
                c_c3 = 4 * Side + 3
                
                # Define the 14 bilinear forms
                # a = A[a_r0, a_c0]
                # b = A[a_r0, a_c1]
                # c = A[a_r1, a_c0]
                # d = A[a_r1, a_c1]
                
                # e = B[b_r0, b_c0]
                # f = B[b_r0, b_c1]
                # g = B[b_r1, b_c0]
                # h = B[b_r1, b_c1]
                
                # e' = B[b_r0, b_c2]
                # f' = B[b_r0, b_c3]
                # g' = B[b_r1, b_c2]
                # h' = B[b_r1, b_c3]
                
                # M_L1: a(e+h)
                U[a_r0 * m + a_c0][col] += 1
                V[b_r0 * p + b_c0][col] += 1
                V[b_r1 * p + b_c1][col] += 1
                # W contributions for M_L1: c00 += 1, c11 += 1
                W[c_c0 * n + c_r0][col] += 1
                W[c_c1 * n + c_r1][col] += 1
                col += 1
                
                # M_L2: (a+b)(g+h)
                U[a_r0 * m + a_c0][col] += 1
                U[a_r0 * m + a_c1][col] += 1
                V[b_r1 * p + b_c0][col] += 1
                V[b_r1 * p + b_c1][col] += 1
                # W contributions: c01 += 1
                W[c_c1 * n + c_r0][col] += 1
                col += 1
                
                # M_L3: (c+d)(e+f)
                U[a_r1 * m + a_c0][col] += 1
                U[a_r1 * m + a_c1][col] += 1
                V[b_r0 * p + b_c0][col] += 1
                V[b_r0 * p + b_c1][col] += 1
                # W contributions: c10 += 1, c11 += -1
                W[c_c0 * n + c_r1][col] += 1
                W[c_c1 * n + c_r1][col] += -1
                col += 1
                
                # M_L4: d(f-g)
                U[a_r1 * m + a_c1][col] += 1
                V[b_r0 * p + b_c1][col] += 1
                V[b_r1 * p + b_c0][col] += -1
                # W contributions: c00 += 1, c01 += -1
                W[c_c0 * n + c_r0][col] += 1
                W[c_c1 * n + c_r0][col] += -1
                col += 1
                
                # M_L5: (a+c)e
                U[a_r0 * m + a_c0][col] += 1
                U[a_r1 * m + a_c0][col] += 1
                V[b_r0 * p + b_c0][col] += 1
                # W contributions: c10 += -1, c11 += 1
                W[c_c0 * n + c_r1][col] += -1
                W[c_c1 * n + c_r1][col] += 1
                col += 1
                
                # M_L6: (b-d)h
                U[a_r0 * m + a_c1][col] += 1
                U[a_r1 * m + a_c1][col] += -1
                V[b_r1 * p + b_c1][col] += 1
                # W contributions: c00 += -1, c01 += 1
                W[c_c0 * n + c_r0][col] += -1
                W[c_c1 * n + c_r0][col] += 1
                col += 1
                
                # M_L7: (c+b)(f-h)
                U[a_r1 * m + a_c0][col] += 1
                U[a_r0 * m + a_c1][col] += 1
                V[b_r0 * p + b_c1][col] += 1
                V[b_r1 * p + b_c1][col] += -1
                # W contributions: c00 += 1, c10 += -1
                W[c_c0 * n + c_r0][col] += 1
                W[c_c0 * n + c_r1][col] += -1
                col += 1
                
                # Now R side
                # M_R1: a(e'+h')
                U[a_r0 * m + a_c0][col] += 1
                V[b_r0 * p + b_c2][col] += 1
                V[b_r1 * p + b_c3][col] += 1
                # W contributions: c02 += 1, c13 += 1
                W[c_c2 * n + c_r0][col] += 1
                W[c_c3 * n + c_r1][col] += 1
                col += 1
                
                # M_R2: (a+b)(g'+h')
                U[a_r0 * m + a_c0][col] += 1
                U[a_r0 * m + a_c1][col] += 1
                V[b_r1 * p + b_c2][col] += 1
                V[b_r1 * p + b_c3][col] += 1
                # W contributions: c03 += 1
                W[c_c3 * n + c_r0][col] += 1
                col += 1
                
                # M_R3: (c+d)(e'+f')
                U[a_r1 * m + a_c0][col] += 1
                U[a_r1 * m + a_c1][col] += 1
                V[b_r0 * p + b_c2][col] += 1
                V[b_r0 * p + b_c3][col] += 1
                # W contributions: c12 += 1, c13 += -1
                W[c_c2 * n + c_r1][col] += 1
                W[c_c3 * n + c_r1][col] += -1
                col += 1
                
                # M_R4: d(f'-g')
                U[a_r1 * m + a_c1][col] += 1
                V[b_r0 * p + b_c3][col] += 1
                V[b_r1 * p + b_c2][col] += -1
                # W contributions: c02 += 1, c03 += -1
                W[c_c2 * n + c_r0][col] += 1
                W[c_c3 * n + c_r0][col] += -1
                col += 1
                
                # M_R5: (a+c)e'
                U[a_r0 * m + a_c0][col] += 1
                U[a_r1 * m + a_c0][col] += 1
                V[b_r0 * p + b_c2][col] += 1
                # W contributions: c12 += -1, c13 += 1
                W[c_c2 * n + c_r1][col] += -1
                W[c_c3 * n + c_r1][col] += 1
                col += 1
                
                # M_R6: (b-d)h'
                U[a_r0 * m + a_c1][col] += 1
                U[a_r1 * m + a_c1][col] += -1
                V[b_r1 * p + b_c3][col] += 1
                # W contributions: c02 += -1, c03 += 1
                W[c_c2 * n + c_r0][col] += -1
                W[c_c3 * n + c_r0][col] += 1
                col += 1
                
                # M_R7: (c+b)(f'-h')
                U[a_r1 * m + a_c0][col] += 1
                U[a_r0 * m + a_c1][col] += 1
                V[b_r0 * p + b_c3][col] += 1
                V[b_r1 * p + b_c3][col] += -1
                # W contributions: c02 += 1, c12 += -1
                W[c_c2 * n + c_r0][col] += 1
                W[c_c2 * n + c_r1][col] += -1
                col += 1

    return {"U": U, "V": V, "W": W}
