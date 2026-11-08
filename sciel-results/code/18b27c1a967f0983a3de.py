import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # A: 2x4, B: 4x4, C: 2x4
    
    # Strassen 2x2 decomposition data
    # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    # M1..M7
    U_local = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 0, 1, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, -1, 0, 1]  # M7
    ]
    V_local = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 0, 1, -1], # M3
        [-1, 1, 0, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 0, 1, 0],  # M6
        [0, 1, 0, 1]   # M7
    ]
    W_local = [
        [1, 0, 0, 1, -1, 0, 1],  # z00
        [0, 0, 0, 0, 1, 1, 0],   # z01
        [0, 1, 1, 0, 0, 0, 0],   # z10
        [1, -1, 1, 1, 0, 0, 0]   # z11
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)] # 8 x 28
    V = [[0] * rank for _ in range(m * p)] # 16 x 28
    W = [[0] * rank for _ in range(p * n)] # 8 x 28
    
    # Define mappings for each sub-problem
    # SP1: A1 B11 -> CL
    # A1: cols 0,1. Global a = 4i + k. Local a' = 2i + k_loc. k_loc = k.
    map_A1 = {0: 0, 1: 1, 4: 2, 5: 3}
    # B11: rows 0,1 cols 0,1. Global b = 4k + l. Local b' = 2k_loc + l_loc.
    map_B11 = {0: 0, 1: 1, 4: 2, 5: 3}
    # CL: cols 0,1. Global c = l*n + i = 2l + i.
    # Local c' order: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) in (i,l)
    # c=0 (l=0,i=0) -> 0
    # c=2 (l=1,i=0) -> 1
    # c=1 (l=0,i=1) -> 2
    # c=3 (l=1,i=1) -> 3
    map_CL = {0: 0, 2: 1, 1: 2, 3: 3}
    
    # SP2: A2 B21 -> CL
    # A2: cols 2,3. Global a = 4i + k. Local a' = 2i + k_loc. k_loc = k-2.
    map_A2 = {2: 0, 3: 1, 6: 2, 7: 3}
    # B21: rows 2,3 cols 0,1. Global b = 4k + l. Local b' = 2k_loc + l_loc. k_loc = k-2.
    map_B21 = {8: 0, 9: 1, 12: 2, 13: 3}
    # CL: same as SP1
    map_CL2 = map_CL
    
    # SP3: A1 B12 -> CR
    # A1: same as SP1
    map_A1_3 = map_A1
    # B12: rows 0,1 cols 2,3. Global b = 4k + l. Local b' = 2k_loc + l_loc. l_loc = l-2.
    map_B12 = {2: 0, 3: 1, 6: 2, 7: 3}
    # CR: cols 2,3. Global c = 2l + i.
    # l=2: c=4(i=0), c=5(i=1)
    # l=3: c=6(i=0), c=7(i=1)
    # Local 0: (i=0, l=2) -> c=4
    # Local 1: (i=0, l=3) -> c=6
    # Local 2: (i=1, l=2) -> c=5
    # Local 3: (i=1, l=3) -> c=7
    map_CR = {4: 0, 6: 1, 5: 2, 7: 3}
    
    # SP4: A2 B22 -> CR
    # A2: same as SP2
    map_A2_4 = map_A2
    # B22: rows 2,3 cols 2,3. Global b = 4k + l. Local b' = 2k_loc + l_loc. k_loc=k-2, l_loc=l-2.
    map_B22 = {10: 0, 11: 1, 14: 2, 15: 3}
    # CR: same as SP3
    map_CR4 = map_CR
    
    # Fill U, V, W
    # col offset for each SP
    sp_configs = [
        (map_A1, map_B11, map_CL, 0),
        (map_A2, map_B21, map_CL2, 7),
        (map_A1_3, map_B12, map_CR, 14),
        (map_A2_4, map_B22, map_CR4, 21)
    ]
    
    for map_A, map_B, map_C, offset in sp_configs:
        for t in range(7):
            col = offset + t
            # Fill U
            for g_idx, l_idx in map_A.items():
                U[g_idx][col] = U_local[t][l_idx]
            # Fill V
            for g_idx, l_idx in map_B.items():
                V[g_idx][col] = V_local[t][l_idx]
            # Fill W
            for g_idx, l_idx in map_C.items():
                W[g_idx][col] = W_local[l_idx][t]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
