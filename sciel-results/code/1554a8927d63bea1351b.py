import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 steps: (alpha, beta, gamma)
    # Each is a dict mapping (i,j) -> coeff
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Part 1: 4x4x4 multiplication (cols 0-3 of B, cols 0-3 of C)
    # Rank 49 = 7 * 7
    for s1 in range(7):
        a1, b1, c1 = strassen_steps[s1]
        for s2 in range(7):
            a2, b2, c2 = strassen_steps[s2]
            col = s1 * 7 + s2
            
            # U: A is 4x4, index = row * m + col
            for (bi, bj), va in a1.items():
                for (si, sj), va2 in a2.items():
                    r = (bi * 2 + si) * m + (bj * 2 + sj)
                    U[r][col] += va * va2
            
            # V: B1 is 4x4 (cols 0-3 of B), index = row * p + col
            for (bi, bj), vb in b1.items():
                for (si, sj), vb2 in b2.items():
                    r = (bi * 2 + si) * p + (bj * 2 + sj)
                    V[r][col] += vb * vb2
            
            # W: C1 is 4x4 (cols 0-3 of C), index = col * n + row
            for (bi, bj), vc in c1.items():
                for (si, sj), vc2 in c2.items():
                    r = (bj * 2 + sj) * n + (bi * 2 + si)
                    W[r][col] += vc * vc2
    
    # Part 2: 4x4x2 multiplication (cols 4-5 of B, cols 4-5 of C)
    # A is 4x4, B2 is 4x2, C2 is 4x2
    # We split A into 4 blocks of 2x2: A00, A01, A10, A11
    # B2 is split into 2 blocks of 2x2: B0 (rows 0-1), B1 (rows 2-3)
    # C2 is split into 2 blocks of 2x2: C0 (rows 0-1), C1 (rows 2-3)
    # 
    # Multiplications:
    # C0 = A00 * B0 + A01 * B1
    # C1 = A10 * B0 + A11 * B1
    #
    # We do 4 separate 2x2x2 multiplications:
    # M1: A00 * B0 -> contributes to C0
    # M2: A01 * B1 -> contributes to C0
    # M3: A10 * B0 -> contributes to C1
    # M4: A11 * B1 -> contributes to C1
    #
    # Each 2x2x2 has rank 7, so total 4 * 7 = 28
    
    mults = [
        # (a_block, b_block, c_block)
        # a_block: (bi, bj) in A
        # b_block: bi in B2 (0 or 1)
        # c_block: bi in C2 (0 or 1)
        ((0, 0), 0, 0),  # A00 * B0 -> C0
        ((0, 1), 1, 0),  # A01 * B1 -> C0
        ((1, 0), 0, 1),  # A10 * B0 -> C1
        ((1, 1), 1, 1),  # A11 * B1 -> C1
    ]
    
    for m_idx, (a_block, b_bi, c_bi) in enumerate(mults):
        a_bi, a_bj = a_block
        for s2 in range(7):
            a2, b2, c2 = strassen_steps[s2]
            col = 49 + m_idx * 7 + s2
            
            # U: A block (a_bi, a_bj)
            for (si, sj), va in a2.items():
                r = (a_bi * 2 + si) * m + (a_bj * 2 + sj)
                U[r][col] += va
            
            # V: B2 block (b_bi, 0) -> rows b_bi*2 to b_bi*2+1, cols 4-5
            for (si, sj), vb in b2.items():
                r = (b_bi * 2 + si) * p + (4 + sj)
                V[r][col] += vb
            
            # W: C2 block (c_bi, 0) -> rows c_bi*2 to c_bi*2+1, cols 4-5
            # W index = col * n + row
            for (si, sj), vc in c2.items():
                r = (4 + sj) * n + (c_bi * 2 + si)
                W[r][col] += vc
    
    return {"U": U, "V": V, "W": W}
