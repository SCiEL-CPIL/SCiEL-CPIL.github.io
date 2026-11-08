import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS
    
    # We model M_{3,4,8} as M_{3,2,4} \otimes M_{1,2,2}
    # Rank(M_{1,2,2}) = 4 (trivial)
    # Rank(M_{3,2,4}) = 22 (via 1x2x4 rank 8 + 2x2x4 rank 14)
    # Total Rank = 22 * 4 = 88
    
    rank_total = 88
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Decomposition of M_{1,2,2} (Rank 4)
    # i_A=1, j_A=2, k_A=2
    # U_A: 1x4, V_A: 2x4, W_A: 2x4
    U_A = [[1, 1, 1, 1]]
    V_A = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_A = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    
    # Decomposition of M_{3,2,4} (Rank 22)
    # Split i_B (dim 3) into 1 (i=0) and 2 (i=1,2)
    # Part 1: M_{1,2,4} (Rank 8)
    # Part 2: M_{2,2,4} (Rank 14) -> Two M_{2,2,2} Strassen blocks
    
    # U_B: 3x22, V_B: 2x22, W_B: 4x22
    U_B = [[0] * 22 for _ in range(3)]
    V_B = [[0] * 22 for _ in range(2)]
    W_B = [[0] * 22 for _ in range(4)]
    
    col_B = 0
    
    # Part 1: i_B=0, M_{1,2,4}
    # j_B in 0..1, k_B in 0..3
    for j_B in range(2):
        for k_B in range(4):
            U_B[0][col_B] = 1
            V_B[j_B][col_B] = 1
            W_B[k_B][col_B] = 1
            col_B += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
    # Part 2: i_B in 1..2, M_{2,2,4}
    # Split k_B into two blocks of 2: [0,1] and [2,3]
    # Each is M_{2,2,2} Strassen (Rank 7)
    
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for k_block in range(2):
        k_start = k_block * 2
        for t in range(7):
            # U_B: i_B in 1..2 (local 0..1)
            for l in range(4):
                if strassen_U[t][l] != 0:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    gj = lj
                    U_B[gi][col_B] += strassen_U[t][l]
                    
            # V_B: j_B in 0..1 (local 0..1)
            for l in range(4):
                if strassen_V[t][l] != 0:
                    lj = l // 2
                    lk = l % 2
                    gj = lj
                    gk = lk # local k
                    # Note: V_B is for M_{2,2,4}, so V_B[j_B, col]
                    # The local j_B is 0..1.
                    V_B[gj][col_B] += strassen_V[t][l]
                    
            # W_B: k_B in 0..3
            for l in range(4):
                if strassen_W[t][l] != 0:
                    li = l // 2
                    lk = l % 2
                    gi = li # local i (0..1) -> maps to i_B 1..2? No, W is output C.
                    # W_B[k_B, col]
                    gk = k_start + lk
                    W_B[gk][col_B] += strassen_W[t][l]
                    
            col_B += 1
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
                
    # Now combine M_{3,2,4} and M_{1,2,2} via Kronecker product
    # Total rank r = r_B * 4 + r_A
    # i = i_B (since i_A=1)
    # j = j_B * 2 + j_A
    # k = k_B * 2 + k_A
    
    for r_B in range(22):
        for r_A in range(4):
            r = r_B * 4 + r_A
            
            # U[i, r] = U_B[i_B, r_B] * U_A[i_A, r_A]
            # i = i_B
            for i_B in range(3):
                val = U_B[i_B][r_B] * U_A[0][r_A]
                if val != 0:
                    U[i_B * m + (0)][r] += val # Wait, j index?
                    # j = j_B * 2 + j_A
                    # U is (n*m) x rank. n=3, m=4.
                    # U[i, j, r].
                    # i is 0..2. j is 0..3.
                    # j = j_B * 2 + j_A.
                    # j_B is 0..1, j_A is 0..1.
                    # We need to iterate j_B, j_A?
                    # No, U_B is for M_{3,2,4}. U_B[i_B, j_B, r_B].
                    # U_A is for M_{1,2,2}. U_A[i_A, j_A, r_A].
                    # My U_B above was 3x22. It missed j_B dimension!
                    # M_{3,2,4} has dims 3x2x4.
                    # U_B should be 3x2x22.
                    
    # Re-implementing U_B, V_B, W_B with correct dimensions
    U_B = [[[0] * 22 for _ in range(2)] for _ in range(3)]
    V_B = [[[0] * 22 for _ in range(2)] for _ in range(2)] # V_B[j_B, k_B, r_B]
    W_B = [[[0] * 22 for _ in range(4)] for _ in range(2)] # W_B[i_B, k_B, r_B]
    
    col_B = 0
    # Part 1: i_B=0
    for j_B in range(2):
        for k_B in range(4):
            U_B[0][j_B][col_B] = 1
            V_B[j_B][k_B][col_B] = 1
            W_B[0][k_B][col_B] = 1
            col_B += 1
            
    # Part 2: i_B=1,2
    for k_block in range(2):
        k_start = k_block * 2
        for t in range(7):
            for l in range(4):
                if strassen_U[t][l] != 0:
                    li = l // 2
                    lj = l % 2
                    gi = 1 + li
                    U_B[gi][lj][col_B] += strassen_U[t][l]
                    
            for l in range(4):
                if strassen_V[t][l] != 0:
                    lj = l // 2
                    lk = l % 2
                    V_B[lj][k_start + lk][col_B] += strassen_V[t][l]
                    
            for l in range(4):
                if strassen_W[t][l] != 0:
                    li = l // 2
                    lk = l % 2
                    W_B[1 + li][k_start + lk][col_B] += strassen_W[t][l]
                    
            col_B += 1
            
    # Combine
    for r_B in range(22):
        for r_A in range(4):
            r = r_B * 4 + r_A
            for i_B in range(3):
                for j_B in range(2):
                    for j_A in range(2):
                        for k_B in range(4):
                            for k_A in range(2):
                                i = i_B
                                j = j_B * 2 + j_A
                                k = k_B * 2 + k_A
                                
                                u_val = U_B[i_B][j_B][r_B] * U_A[0][r_A]
                                if u_val != 0:
                                    U[i * m + j][r] += u_val
                                    
                                v_val = V_B[j_B][k_B][r_B] * V_A[j_A][r_A]
                                if v_val != 0:
                                    V[j * p + k][r] += v_val
                                    
                                w_val = W_B[i_B][k_B][r_B] * W_A[k_A][r_A]
                                if w_val != 0:
                                    W[k * n + i][r] += w_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
