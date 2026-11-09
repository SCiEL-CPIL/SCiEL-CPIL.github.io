"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 4, 4, 5
    # Rank 72 decomposition using Strassen for 2x2x2 blocks and schoolbook for 2x2x1 blocks
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define index partitions
    I1 = [0, 1]
    I2 = [2, 3]
    J1 = [0, 1]
    J2 = [2, 3]
    K1 = [0, 1]
    K2 = [2, 3]
    K3 = [4]
    
    # Strassen factors for 2x2x2
    # M1
    S_M1_U = [(0, 0, 1), (1, 1, 1)]
    S_M1_V = [(0, 0, 1), (1, 1, 1)]
    S_M1_W = [(0, 0, 1), (1, 1, 1)]
    
    # M2
    S_M2_U = [(1, 0, 1), (1, 1, 1)]
    S_M2_V = [(0, 0, 1)]
    S_M2_W = [(0, 1, 1), (1, 1, -1)]
    
    # M3
    S_M3_U = [(0, 0, 1)]
    S_M3_V = [(1, 0, 1), (1, 1, -1)]
    S_M3_W = [(0, 1, 1), (1, 1, 1)]
    
    # M4
    S_M4_U = [(1, 1, 1)]
    S_M4_V = [(0, 1, 1), (0, 0, -1)]
    S_M4_W = [(0, 0, 1), (1, 0, 1)]
    
    # M5
    S_M5_U = [(0, 0, 1), (0, 1, 1)]
    S_M5_V = [(1, 1, 1)]
    S_M5_W = [(0, 0, -1), (0, 1, 1)]
    
    # M6
    S_M6_U = [(1, 0, 1), (0, 0, -1)]
    S_M6_V = [(0, 0, 1), (0, 1, 1)]
    S_M6_W = [(1, 1, 1)]
    
    # M7
    S_M7_U = [(0, 1, 1), (1, 1, -1)]
    S_M7_V = [(1, 0, 1), (1, 1, 1)]
    S_M7_W = [(0, 0, 1)]
    
    strassen_blocks = [
        (I1, J1, K1), (I1, J1, K2),
        (I1, J2, K1), (I1, J2, K2),
        (I2, J1, K1), (I2, J1, K2),
        (I2, J2, K1), (I2, J2, K2)
    ]
    
    for I_set, J_set, K_set in strassen_blocks:
        # M1
        for (li, lj, val) in S_M1_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M1_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M1_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M2
        for (li, lj, val) in S_M2_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M2_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M2_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M3
        for (li, lj, val) in S_M3_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M3_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M3_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M4
        for (li, lj, val) in S_M4_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M4_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M4_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M5
        for (li, lj, val) in S_M5_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M5_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M5_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M6
        for (li, lj, val) in S_M6_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M6_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M6_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1
        
        # M7
        for (li, lj, val) in S_M7_U:
            U[I_set[li] * m + J_set[lj]][col] += val
        for (lj, lk, val) in S_M7_V:
            V[J_set[lj] * p + K_set[lk]][col] += val
        for (lk, li, val) in S_M7_W:
            W[K_set[lk] * n + I_set[li]][col] += val
        col += 1

    # Schoolbook blocks for 2x2x1
    schoolbook_blocks = [
        (I1, J1, K3), (I1, J2, K3),
        (I2, J1, K3), (I2, J2, K3)
    ]
    
    for I_set, J_set, K_set in schoolbook_blocks:
        for li in range(2):
            for lj in range(2):
                # U
                U[I_set[li] * m + J_set[lj]][col] = 1
                # V
                V[J_set[lj] * p + K_set[0]][col] = 1
                # W
                W[K_set[0] * n + I_set[li]][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
