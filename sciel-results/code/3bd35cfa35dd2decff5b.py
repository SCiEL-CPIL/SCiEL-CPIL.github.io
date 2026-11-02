"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28  # 4 blocks * 7 (Strassen rank)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen decomposition for 2x2x2
    # Indices for local U (A matrix): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for local V (B matrix): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for local W (C matrix): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Note: W index is k*2 + i.
    # C00 -> 0, C01 -> 1 (k=0, i=1? No, k*2+i. C01 is i=0, k=1? 
    # Let's re-verify W index mapping.
    # Task says: W[k*n+i, r]. n=2.
    # C[i, k].
    # If i=0, k=0 -> index 0*2+0=0.
    # If i=0, k=1 -> index 1*2+0=2.
    # If i=1, k=0 -> index 0*2+1=1.
    # If i=1, k=1 -> index 1*2+1=3.
    # So local W indices:
    # C00 (i=0,k=0) -> 0
    # C10 (i=1,k=0) -> 1
    # C01 (i=0,k=1) -> 2
    # C11 (i=1,k=1) -> 3
    
    # My previous W vectors:
    # w1: C00, C11 -> 1, 0, 0, 1. Correct.
    # w2: C10, -C11 -> 0, 1, 0, -1. Correct.
    # w3: C01, C11 -> 0, 0, 1, 1. Correct.
    # w4: C00, C10 -> 1, 1, 0, 0. Correct.
    # w5: -C00, C01 -> -1, 0, 1, 0. Correct.
    # w6: C11 -> 0, 0, 0, 1. Correct.
    # w7: C00 -> 1, 0, 0, 0. Correct.
    
    local_U = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    local_V = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    
    local_W = [
        [1, 0, 0, 1], # P1
        [0, 1, 0, -1],# P2
        [0, 0, 1, 1], # P3
        [1, 1, 0, 0], # P4
        [-1, 0, 1, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # 4 blocks: (j_block, k_block)
    # j_block 0 -> j in {0,1}
    # j_block 1 -> j in {2,3}
    # k_block 0 -> k in {0,1}
    # k_block 1 -> k in {2,3}
    
    for jb in range(2):
        for kb in range(2):
            block_id = jb * 2 + kb
            base_col = block_id * 7
            
            for r in range(7):
                col = base_col + r
                
                # Fill U
                for u_idx in range(4):
                    u_val = local_U[r][u_idx]
                    if u_val != 0:
                        i_local = u_idx // 2
                        j_local = u_idx % 2
                        i_global = i_local
                        j_global = jb * 2 + j_local
                        u_row = i_global * m + j_global
                        U[u_row][col] = u_val
                
                # Fill V
                for v_idx in range(4):
                    v_val = local_V[r][v_idx]
                    if v_val != 0:
                        j_local = v_idx // 2
                        k_local = v_idx % 2
                        j_global = jb * 2 + j_local
                        k_global = kb * 2 + k_local
                        v_row = j_global * p + k_global
                        V[v_row][col] = v_val
                
                # Fill W
                for w_idx in range(4):
                    w_val = local_W[r][w_idx]
                    if w_val != 0:
                        # w_idx is k_local * 2 + i_local
                        i_local = w_idx % 2
                        k_local = w_idx // 2
                        i_global = i_local
                        k_global = kb * 2 + k_local
                        w_row = k_global * n + i_global
                        W[w_row][col] = w_val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
