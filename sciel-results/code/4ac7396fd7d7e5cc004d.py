import time

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen base matrices for 2x2x2
    # Local indices 0..3 for U, V, W
    # U rows: 0:11, 1:12, 2:21, 3:22
    # V rows: 0:11, 1:12, 2:21, 3:22
    # W rows: 0:11, 1:12, 2:21, 3:22
    
    # Define the 7 columns of the base Strassen decomposition
    # col: (U_entries, V_entries, W_entries)
    # U_entries: dict {local_row: val}
    # V_entries: dict {local_row: val}
    # W_entries: dict {local_row: val}
    
    strassen_cols = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]
    
    # Define the 4 blocks
    # Block 1: A_L (U rows 0,1,4,5) * B_00 (V rows 0,1,4,5) -> C_00 (W rows 0,1,2,3)
    # Block 2: A_L (U rows 0,1,4,5) * B_01 (V rows 2,3,6,7) -> C_01 (W rows 4,5,6,7)
    # Block 3: A_R (U rows 2,3,6,7) * B_10 (V rows 8,9,12,13) -> C_00 (W rows 0,1,2,3)
    # Block 4: A_R (U rows 2,3,6,7) * B_11 (V rows 10,11,14,15) -> C_01 (W rows 4,5,6,7)
    
    # Mapping local 0..3 to global indices
    # Local 0 (11) -> Global index depends on block
    # For U (A matrix, 2x4):
    # A_L (cols 0,1):
    #   Local 0 (0,0) -> Global 0
    #   Local 1 (0,1) -> Global 1
    #   Local 2 (1,0) -> Global 4
    #   Local 3 (1,1) -> Global 5
    # A_R (cols 2,3):
    #   Local 0 (0,2) -> Global 2
    #   Local 1 (0,3) -> Global 3
    #   Local 2 (1,2) -> Global 6
    #   Local 3 (1,3) -> Global 7
    
    # For V (B matrix, 4x4):
    # B_00 (rows 0,1 cols 0,1):
    #   Local 0 (0,0) -> Global 0
    #   Local 1 (0,1) -> Global 1
    #   Local 2 (1,0) -> Global 4
    #   Local 3 (1,1) -> Global 5
    # B_01 (rows 0,1 cols 2,3):
    #   Local 0 (0,2) -> Global 2
    #   Local 1 (0,3) -> Global 3
    #   Local 2 (1,2) -> Global 6
    #   Local 3 (1,3) -> Global 7
    # B_10 (rows 2,3 cols 0,1):
    #   Local 0 (2,0) -> Global 8
    #   Local 1 (2,1) -> Global 9
    #   Local 2 (3,0) -> Global 12
    #   Local 3 (3,1) -> Global 13
    # B_11 (rows 2,3 cols 2,3):
    #   Local 0 (2,2) -> Global 10
    #   Local 1 (2,3) -> Global 11
    #   Local 2 (3,2) -> Global 14
    #   Local 3 (3,3) -> Global 15
    
    # For W (C matrix, 2x4):
    # C_00 (cols 0,1):
    #   Local 0 (0,0) -> Global 0
    #   Local 1 (0,1) -> Global 1
    #   Local 2 (1,0) -> Global 2
    #   Local 3 (1,1) -> Global 3
    # C_01 (cols 2,3):
    #   Local 0 (0,2) -> Global 4
    #   Local 1 (0,3) -> Global 5
    #   Local 2 (1,2) -> Global 6
    #   Local 3 (1,3) -> Global 7
    
    blocks = [
        # (U_map, V_map, W_map, rank_offset)
        (
            [0, 1, 4, 5], # A_L
            [0, 1, 4, 5], # B_00
            [0, 1, 2, 3], # C_00
            0
        ),
        (
            [0, 1, 4, 5], # A_L
            [2, 3, 6, 7], # B_01
            [4, 5, 6, 7], # C_01
            7
        ),
        (
            [2, 3, 6, 7], # A_R
            [8, 9, 12, 13], # B_10
            [0, 1, 2, 3], # C_00
            14
        ),
        (
            [2, 3, 6, 7], # A_R
            [10, 11, 14, 15], # B_11
            [4, 5, 6, 7], # C_01
            21
        )
    ]
    
    for u_map, v_map, w_map, offset in blocks:
        for i in range(7):
            col = offset + i
            u_entries, v_entries, w_entries = strassen_cols[i]
            
            for local_idx, val in u_entries.items():
                U[u_map[local_idx]][col] += val
                
            for local_idx, val in v_entries.items():
                V[v_map[local_idx]][col] += val
                
            for local_idx, val in w_entries.items():
                W[w_map[local_idx]][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
