DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = 2, 2, 10
    num_blocks = p // 2  # 5 blocks of 2 columns each
    rank_per_block = 7
    rank = num_blocks * rank_per_block  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    # Strassen terms: (a_factors, b_factors, c_factors)
    # Each factor is (index1, index2, sign)
    # a_factors: (i, j, sign) for A[i,j]
    # b_factors: (j, k_local, sign) for B[j, k_local]
    # c_factors: (k_local, i, sign) for C[i, k_local]
    
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # P2 = (A10+A11)B00
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
        # P3 = A00(B11-B10)
        ([(0,0,1)], [(1,1,1), (1,0,-1)], [(1,0,1)]),
        # P4 = A11(B10-B00)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,-1), (0,1,1)]),
        # P5 = (A00+A01)B11
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,1), (1,0,1)]),
        # P6 = (A10-A00)(B00+B01)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # P7 = (A01-A11)(B10+B11)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,1,1)]),
    ]
    
    for block in range(num_blocks):
        for t in range(rank_per_block):
            c = block * rank_per_block + t
            a_factors, b_factors, c_factors = strassen_terms[t]
            
            for (i, j, sign) in a_factors:
                U[i * m + j][c] += sign
            
            for (j, k_local, sign) in b_factors:
                k_global = block * 2 + k_local
                V[j * p + k_global][c] += sign
            
            for (k_local, i, sign) in c_factors:
                k_global = block * 2 + k_local
                W[k_global * n + i][c] += sign
    
    return {"U": U, "V": V, "W": W}
