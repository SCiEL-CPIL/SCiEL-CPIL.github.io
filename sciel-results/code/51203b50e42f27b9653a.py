"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    rank = 46
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # Define 23 terms for 3x3x3 block 0 (k=0,1,2) and block 1 (k=3,4,5)
    # I'll use a known rank-23 decomposition for 3x3x3.
    # Here are the 23 terms (u, v, w) for 3x3x3, where:
    # u[i*3+j], v[j*3+k], w[k*3+i] for i,j,k in 0,1,2.
    
    # I'll define the 23 terms as a list of (u_dict, v_dict, w_dict)
    # where each dict maps index to coefficient.
    
    terms_3x3x3 = []
    
    # Strassen's 7 for the 2x2x2 sub-block (i,j,k in 0,1)
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    u1 = {0: 1, 4: 1}  # A00, A11
    v1 = {0: 1, 3: 1}  # B00, B11
    w1 = {0: 1, 4: 1}  # C00, C11 (w[k*3+i]: C00 is k=0,i=0 -> 0; C11 is k=1,i=1 -> 1*3+1=4)
    terms_3x3x3.append((u1, v1, w1))
    
    # M2 = (A10+A11)B00 -> C10
    u2 = {3: 1, 4: 1}  # A10, A11
    v2 = {0: 1}  # B00
    w2 = {3: 1}  # C10: k=0,i=1 -> 0*3+1=1? Wait, C[i,k], so C10 is i=1,k=0. W index is k*3+i = 0*3+1=1.
    # Correction: W index is k*n+i = k*3+i. For C[i,k], the W index is k*3+i.
    # C10: i=1, k=0 -> W index = 0*3+1 = 1.
    # C00: i=0, k=0 -> 0
    # C11: i=1, k=1 -> 1*3+1=4
    # C01: i=0, k=1 -> 1*3+0=3
    # So w1 should be {0: 1, 4: 1} for C00 and C11. Correct.
    # w2 for C10: i=1, k=0 -> 0*3+1=1. So w2={1:1}.
    w2 = {1: 1}
    terms_3x3x3.append((u2, v2, w2))
    
    # M3 = A00(B10-B11) -> C01
    u3 = {0: 1}  # A00
    v3 = {3: 1, 4: -1}  # B10, B11
    w3 = {3: 1}  # C01: i=0,k=1 -> 1*3+0=3
    terms_3x3x3.append((u3, v3, w3))
    
    # M4 = A11(B01-B00) -> C10
    u4 = {4: 1}  # A11
    v4 = {1: 1, 0: -1}  # B01, B00
    w4 = {1: 1}  # C10: i=1,k=0 -> 1
    terms_3x3x3.append((u4, v4, w4))
    
    # M5 = (A00+A01)B11 -> C01
    u5 = {0: 1, 1: 1}  # A00, A01
    v5 = {4: 1}  # B11
    w5 = {3: 1}  # C01: 3
    terms_3x3x3.append((u5, v5, w5))
    
    # M6 = (A10-A00)(B00+B01) -> C10
    u6 = {3: 1, 0: -1}  # A10, A00
    v6 = {0: 1, 1: 1}  # B00, B01
    w6 = {1: 1}  # C10: 1
    terms_3x3x3.append((u6, v6, w6))
    
    # M7 = (A01-A11)(B10+B11) -> C01
    u7 = {1: 1, 4: -1}  # A01, A11
    v7 = {3: 1, 4: 1}  # B10, B11
    w7 = {3: 1}  # C01: 3
    terms_3x3x3.append((u7, v7, w7))
    
    # Now 16 more terms for the index-2 parts.
    # I'll use schoolbook for the remaining 19 terms, but I need only 16.
    # This won't work.
    
    # Given the complexity, I'll abort and use schoolbook for the whole thing.
    
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
