"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2, 2, 1
    # Split p=6 into 2, 2, 2
    # Blocks:
    # (0,0): 2x2x2 Strassen (Rank 7)
    # (0,1): 2x2x2 Strassen (Rank 7)
    # (0,2): 2x2x2 Strassen (Rank 7)
    # (1,0): 2x2x2 Strassen (Rank 7)
    # (1,1): 2x2x2 Strassen (Rank 7)
    # (1,2): 2x2x2 Strassen (Rank 7)
    # (2,0): 2x1x2 Schoolbook (Rank 2)
    # (2,1): 2x1x2 Schoolbook (Rank 2)
    # (2,2): 2x1x2 Schoolbook (Rank 2)
    
    total_rank = 6 * 7 + 3 * 2
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    rank_offset = 0
    
    # Strassen 2x2x2 terms
    # (u_indices, v_indices, w_indices, coeffs)
    # We need to map local indices to global.
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal rank_offset
        # 7 terms
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}),
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (0,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (0,0):-1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
        ]
        
        for t in terms:
            u_dict, v_dict, w_dict = t
            col = rank_offset
            for (i, j), val in u_dict.items():
                U[i_off + i, j_off + j][col] = val
            for (j, k), val in v_dict.items():
                V[j_off + j, k_off + k][col] = val
            for (k, i), val in w_dict.items():
                W[k_off + k, i_off + i][col] = val
            rank_offset += 1

    def add_schoolbook_212(i_off, j_off, k_off):
        nonlocal rank_offset
        # A is 2x1, B is 1x2
        # C00 = A00*B00
        # C01 = A00*B01
        # C10 = A10*B00
        # C11 = A10*B01
        terms = [
            ({(0,0):1}, {(0,0):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1}, {(0,1):1}),
            ({(1,0):1}, {(0,0):1}, {(1,0):1}),
            ({(1,0):1}, {(0,1):1}, {(1,1):1})
        ]
        for t in terms:
            u_dict, v_dict, w_dict = t
            col = rank_offset
            for (i, j), val in u_dict.items():
                U[i_off + i, j_off + j][col] = val
            for (j, k), val in v_dict.items():
                V[j_off + j, k_off + k][col] = val
            for (k, i), val in w_dict.items():
                W[k_off + k, i_off + i][col] = val
            rank_offset += 1

    # m splits: 0-1, 2-3, 4
    # p splits: 0-1, 2-3, 4-5
    
    j_splits = [0, 2, 4]
    k_splits = [0, 2, 4]
    
    for i_block in range(3):
        for j_block in range(3):
            i_off = 0 # n is always 2, so i_off is 0
            j_off = j_splits[j_block]
            k_off = k_splits[i_block] # Wait, i_block is for m? No.
            
    # Correct loop:
    # m_blocks: 0, 1, 2
    # p_blocks: 0, 1, 2
    
    for mb in range(3):
        for pb in range(3):
            j_off = j_splits[mb]
            k_off = k_splits[pb]
            
            if mb < 2 and pb < 3: # m_len=2
                if pb == 2: # p_len=2 (last block)
                    # Actually p_splits are 2,2,2. So all are len 2.
                    pass
                # All m<2 blocks are 2x2x2
                add_strassen(0, j_off, k_off)
            else: # mb == 2, m_len=1
                # 2x1x2
                add_schoolbook_212(0, j_off, k_off)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
