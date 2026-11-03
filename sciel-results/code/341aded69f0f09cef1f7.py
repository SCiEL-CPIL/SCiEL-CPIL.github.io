import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_school, "V": V_school, "W": W_school}

    # Try Strassen 4x4x4 twice (for 4x4x8)
    # Rank 49 for each 4x4x4
    # Total rank 98
    
    # Define Strassen 2x2 factors
    # U2 shape (4, 7), V2 shape (4, 7), W2 shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a+b)(h)
    # M4 = (d-b)(f+h)
    # M5 = (a-d)(e+f)
    # M6 = (b-c)(e+h)
    # M7 = (a+c)(f-h)
    
    # U2
    U2 = [
        [1, 0, 1, 0, 1, 0, 1], # a
        [0, 0, 1, -1, 0, 1, 0], # b
        [0, 1, 0, 0, 0, -1, 1], # c
        [1, 1, 0, 1, -1, 0, 0]  # d
    ]
    # V2
    V2 = [
        [1, 1, 0, 0, 1, 1, 0], # e
        [0, 0, 0, 1, 1, 0, 1], # f
        [0, 0, 0, 0, 0, 0, 0], # g (Wait, g is not used in these formulas? No, that's wrong)
        [1, 0, 1, 1, 0, 1, -1] # h
    ]
    # The formulas I used earlier didn't use g (B[1,0]).
    # This means the Strassen formulas I recalled are for a specific alignment or I am missing terms.
    # Standard Strassen uses all 4 elements of B.
    # My V2[2] is all zeros. This is incorrect.
    
    # Correct Strassen V2:
    # M1: e, h
    # M2: e
    # M3: h
    # M4: f, h
    # M5: e, f
    # M6: e, h
    # M7: f, h
    # Still no g.
    
    # The issue is that the standard Strassen algorithm assumes B is aligned such that 
    # the terms are grouped. But it must cover all products.
    # Actually, the Strassen algorithm works for any 2x2 matrices.
    # The formulas must involve g.
    # Let's check M4 = (d-b)(f+h). No g.
    # M7 = (a+c)(f-h). No g.
    # It seems my recalled formulas are incomplete or for a different partition.
    
    # I will fallback to schoolbook to ensure correctness.
    return fallback

# EVOLVE-BLOCK-END
