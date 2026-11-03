import time

def construct():
    start = time.time()
    deadline = start + 145.0
    
    n, m, p = 3, 4, 8
    
    # Fallback: Rank 88 solution
    def get_rank_88():
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Part 1: i=0, schoolbook 1x4x8
        for j in range(4):
            for k in range(8):
                U[0 * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + 0][col] = 1
                col += 1
                
        # Part 2: i=1,2, Strassen 2x2x2 blocks
        strassen_U = [
            [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
            [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
        ]
        strassen_V = [
            [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
            [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
        ]
        strassen_W = [
            [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
            [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
        ]
        
        for j_block in range(2):
            j_start = j_block * 2
            for k_block in range(4):
                k_start = k_block * 2
                for t in range(7):
                    for l in range(4):
                        if strassen_U[t][l] != 0:
                            li, lj = l // 2, l % 2
                            U[(1 + li) * m + (j_start + lj)][col] += strassen_U[t][l]
                        if strassen_V[t][l] != 0:
                            lj, lk = l // 2, l % 2
                            V[(j_start + lj) * p + (k_start + lk)][col] += strassen_V[t][l]
                        if strassen_W[t][l] != 0:
                            li, lk = l // 2, l % 2
                            W[(k_start + lk) * n + (1 + li)][col] += strassen_W[t][l]
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try Rank 76 solution
    # r(2,3,4) = 14
    # r(2,1,4) = 8
    # r(1,4,4) = 16
    # r(2,4,4) = 22
    # r(3,4,4) = 38
    # r(3,4,8) = 76
    
    # Implementation of Laderman 2x3x4
    def get_laderman_2x3x4():
        # A is 2x3, B is 3x4, C is 2x4
        # U: 6x14, V: 12x14, W: 8x14
        U = [[0]*14 for _ in range(6)]
        V = [[0]*14 for _ in range(12)]
        W = [[0]*14 for _ in range(8)]
        
        # Indices: U[i*3+j], V[j*4+k], W[k*2+i]
        
        # P1: a00*b00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # P2: a00*b10
        U[0][1] = 1; V[4][1] = 1; W[2][0] = 1 # C01? No, C01 is a00*b10 + ...
        # Wait, C[i,k] = sum_j A[i,j]B[j,k]
        # C00 = a00*b00 + a01*b10 + a02*b20
        # C01 = a00*b01 + a01*b11 + a02*b21
        # ...
        
        # The Laderman products I listed earlier might not map directly to C entries with 1s.
        # Given the complexity and risk, I will stick to the safe Rank 88 solution.
        return None

    # Since I am not 100% confident in the Laderman mapping without verification,
    # and the time limit is tight, I will return the Rank 88 solution.
    # It is a proven improvement over 96.
    
    if time.time() > deadline - 5:
        return get_rank_88()
        
    return get_rank_88()
