"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Tiling strategy:
    # i-groups: {0,1} (size 2), {2} (size 1)
    # j-groups: {0,1} (size 2), {2,3} (size 2), {4} (size 1)
    # k-groups: {0,1} (size 2), {2,3} (size 2), {4} (size 1)
    
    # We will use Strassen for all combinations of i-group {0,1}, j-group {0,1} or {2,3}, k-group {0,1} or {2,3}
    # This gives 4 blocks of 2x2x2.
    
    i_strassen = [0, 1]
    j_strassen_groups = [[0, 1], [2, 3]]
    k_strassen_groups = [[0, 1], [2, 3]]
    
    # Schoolbook for the rest
    # i=2: all j, k
    # i in {0,1}: j=4 or k=4
    
    total_rank = 0
    strassen_blocks = []
    for jg in j_strassen_groups:
        for kg in k_strassen_groups:
            strassen_blocks.append((i_strassen, jg, kg))
            total_rank += 7
            
    # Schoolbook blocks
    # 1. i=2, all j, k
    for i in [2]:
        for j in range(m):
            for k in range(p):
                total_rank += 1
                
    # 2. i in {0,1}, j=4, all k
    for i in i_strassen:
        for j in [4]:
            for k in range(p):
                total_rank += 1
                
    # 3. i in {0,1}, k=4, j in {0,1,2,3} (j=4 already covered)
    for i in i_strassen:
        for j in range(4):
            for k in [4]:
                total_rank += 1
                
    rank = total_rank
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2 block
    # U, V, W are 4-element vectors (flattened 2x2)
    # Index mapping: local (li, lj) -> idx = li*2 + lj
    # U[li, lj], V[lj, lk], W[lk, li]
    
    # M1
    U1 = [1, 0, 0, 1] # (0,0), (1,1)
    V1 = [1, 0, 0, 1] # (0,0), (1,1)
    W1 = [1, 0, 0, 1] # (0,0), (1,1)
    
    # M2
    U2 = [0, 0, 1, 1] # (1,0), (1,1)
    V2 = [1, 0, 0, 0] # (0,0)
    W2 = [0, 0, 1, 0] # (1,0)
    
    # M3
    U3 = [1, 0, 0, 0] # (0,0)
    V3 = [0, 1, 0, -1] # (0,1), (1,1)
    W3 = [0, 1, 0, 1] # (0,1), (1,1)
    
    # M4
    U4 = [0, 0, 0, 1] # (1,1)
    V4 = [-1, 0, 1, 0] # (0,0), (1,0)
    W4 = [1, 0, 1, 0] # (0,0), (1,0)
    
    # M5
    U5 = [1, 1, 0, 0] # (0,0), (0,1)
    V5 = [0, 0, 0, 1] # (1,1)
    W5 = [-1, 1, 0, 0] # (0,0), (0,1)
    
    # M6
    U6 = [-1, 0, 1, 0] # (0,0), (1,0)
    V6 = [1, 1, 0, 0] # (0,0), (0,1)
    W6 = [0, 0, 0, 1] # (1,1)
    
    # M7
    U7 = [0, 1, 0, -1] # (0,1), (1,1)
    V7 = [0, 0, 1, 1] # (1,0), (1,1)
    W7 = [1, 0, 0, 0] # (0,0)
    
    strassen_coeffs = [
        (U1, V1, W1),
        (U2, V2, W2),
        (U3, V3, W3),
        (U4, V4, W4),
        (U5, V5, W5),
        (U6, V6, W6),
        (U7, V7, W7)
    ]
    
    # Fill Strassen blocks
    for (i_list, j_list, k_list) in strassen_blocks:
        i0, i1 = i_list[0], i_list[1]
        j0, j1 = j_list[0], j_list[1]
        k0, k1 = k_list[0], k_list[1]
        
        for r in range(7):
            Ur, Vr, Wr = strassen_coeffs[r]
            
            # U
            for li in range(2):
                for lj in range(2):
                    val = Ur[li * 2 + lj]
                    if val != 0:
                        gi = i0 + li
                        gj = j0 + lj
                        U[gi * m + gj][col + r] += val
                        
            # V
            for lj in range(2):
                for lk in range(2):
                    val = Vr[lj * 2 + lk]
                    if val != 0:
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col + r] += val
                        
            # W
            for lk in range(2):
                for li in range(2):
                    val = Wr[lk * 2 + li]
                    if val != 0:
                        gk = k0 + lk
                        gi = i0 + li
                        W[gk * n + gi][col + r] += val
                        
        col += 7
        
    # Fill Schoolbook blocks
    # 1. i=2
    for i in [2]:
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 2. i in {0,1}, j=4
    for i in i_strassen:
        for j in [4]:
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # 3. i in {0,1}, k=4, j in 0..3
    for i in i_strassen:
        for j in range(4):
            for k in [4]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Verify correctness
    if time.time() < deadline - 5:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err > 1e-9:
            # Fallback to schoolbook if Strassen is wrong
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][c] = 1
                        V[j * p + k][c] = 1
                        W[k * n + i][c] = 1
                        c += 1
            rank = n * m * p
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
