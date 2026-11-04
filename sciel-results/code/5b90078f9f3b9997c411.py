"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Schoolbook rank
    rank_school = n * m * p  # 96
    
    def build_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Strategy:
    # Part 1: Schoolbook for A[0,:] (1x4) * B (4x8): rank 32
    # Part 2: Strassen for A[1:3,:] (2x4) * B (4x8):
    #   Split j into 2 blocks of 2, k into 4 blocks of 2
    #   8 blocks of 2x2x2, each with rank 7
    #   Total for Part 2: 8 * 7 = 56
    # Total rank: 32 + 56 = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Schoolbook for i=0
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
    
    # Check time
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # Strassen's algorithm for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = AB
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U factors (for A): [a, b, c, d]
    strassen_U = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    # V factors (for B): [e, f, g, h]
    strassen_V = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    # W factors (for C): [C11, C12, C21, C22]
    strassen_W = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Part 2: Strassen for i=1,2
    # j blocks: [0,1] and [2,3]
    # k blocks: [0,1], [2,3], [4,5], [6,7]
    
    for j_block in range(2):
        j_start = j_block * 2
        for k_block in range(4):
            k_start = k_block * 2
            
            for t in range(7):
                # U: map local [a,b,c,d] to global A[1:3, j_start:j_start+2]
                for l in range(4):
                    val = strassen_U[t][l]
                    if val != 0:
                        local_i = l // 2  # 0 or 1
                        local_j = l % 2    # 0 or 1
                        global_i = 1 + local_i
                        global_j = j_start + local_j
                        U[global_i * m + global_j][col] += val
                
                # V: map local [e,f,g,h] to global B[j_start:j_start+2, k_start:k_start+2]
                for l in range(4):
                    val = strassen_V[t][l]
                    if val != 0:
                        local_j = l // 2  # 0 or 1
                        local_k = l % 2    # 0 or 1
                        global_j = j_start + local_j
                        global_k = k_start + local_k
                        V[global_j * p + global_k][col] += val
                
                # W: map local [C11,C12,C21,C22] to global C[k_start:k_start+2, 1:3]
                # W index is k*n + i
                for l in range(4):
                    val = strassen_W[t][l]
                    if val != 0:
                        local_k = l // 2  # 0 or 1 (for C11,C12 vs C21,C22)
                        local_i = l % 2    # 0 or 1 (for col 0 vs col 1)
                        global_k = k_start + local_k
                        global_i = 1 + local_i
                        W[global_k * n + global_i][col] += val
                
                col += 1
    
    # Verify
    if col != rank:
        # Fallback to schoolbook
        result = build_schoolbook()
        return result
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
