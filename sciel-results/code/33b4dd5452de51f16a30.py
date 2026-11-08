"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    # We decompose into p//2 = 3 blocks of width 2 in the k dimension
    num_blocks = p // 2  # 3
    # Each block needs 11 rank terms (7 Strassen + 4 scalar)
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 33

    # U: (n*m, rank) = (6, 33)
    # V: (m*p, rank) = (18, 33)
    # W: (p*n, rank) = (12, 33)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Strassen terms for 2x2 * 2x2 multiplication
    # Local indices: A[i,j] with i in 0..1, j in 0..1
    #                B[j,k] with j in 0..1, k in 0..1
    #                C[i,k] with i in 0..1, k in 0..1
    # Local flat indices: A: i*2+j, B: j*2+k, C: i*2+k
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Scalar terms for the cross-block interactions
    # For each block, we need to handle:
    # A[0,2] * B[2, block*2+0] -> C[0, block*2+0]
    # A[0,2] * B[2, block*2+1] -> C[0, block*2+1]
    # A[1,2] * B[2, block*2+0] -> C[1, block*2+0]
    # A[1,2] * B[2, block*2+1] -> C[1, block*2+1]
    
    scalar_terms = [
        # K1: A[0,2] * B[2,0] -> C[0,0]
        ({4: 1}, {4: 1}, {0: 1}),
        # K2: A[0,2] * B[2,1] -> C[0,1]
        ({4: 1}, {5: 1}, {1: 1}),
        # K3: A[1,2] * B[2,0] -> C[1,0]
        ({5: 1}, {4: 1}, {2: 1}),
        # K4: A[1,2] * B[2,1] -> C[1,1]
        ({5: 1}, {5: 1}, {3: 1}),
    ]
    
    # All terms for a block: 7 Strassen + 4 scalar = 11 terms
    all_block_terms = strassen_terms + scalar_terms
    
    for block in range(num_blocks):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(all_block_terms):
            col = block * rank_per_block + t_idx
            
            # Fill U: A indices are global
            # A flat index: i*m + j where i in 0..n-1, j in 0..m-1
            for local_idx, val in u_dict.items():
                # local_idx is i*2+j for j in 0..1, but we need to map to global
                # Actually, the local indices in strassen use j=0,1 which correspond to global j=0,1
                # But wait, the Strassen terms use j=0,1 for the 2x2 block
                # The global A index is i*m + j = i*3 + j
                # Local A index: for i in 0..1, j in 0..1: local = i*2+j
                # So global = i*3 + j
                i_local = local_idx // 2
                j_local = local_idx % 2
                g_idx = i_local * m + j_local
                U[g_idx][col] = val
            
            # Fill V: B indices
            # Local B index: j*2+k where j in 0..1, k in 0..1
            # But for Strassen terms, j=0,1 correspond to global j=0,1
            # For scalar terms, j=2 (the "bottom" row) and k=0,1
            # Global B index: j*p + (block*2 + k)
            for local_idx, val in v_dict.items():
                j_local = local_idx // 2
                k_local = local_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            # Fill W: C indices
            # Convention: W[k*n + i, r]
            # Local C index: i*2+k where i in 0..1, k in 0..1
            # Global C index: (block*2 + k)*n + i
            for local_idx, val in w_dict.items():
                i_local = local_idx // 2
                k_local = local_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
