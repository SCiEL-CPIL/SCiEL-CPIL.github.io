"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use two rank-10 decompositions for M_{2,2,4} blocks
    # Total rank = 10 + 10 = 20? No, that's too optimistic.
    # Actually, let me use a safer approach: keep Strassen but optimize.
    # 
    # Let me try: use Strassen for 2x2x2 (rank 7) on pairs of columns.
    # For 2x2x8, we have 4 such blocks: rank 28.
    #
    # Alternative: Use a single decomposition that handles the 2x8 structure.
    # The matrix multiplication tensor for 2x2 by 2x8 has rank 24.
    #
    # I'll implement the rank-24 decomposition using the following approach:
    # Split B into B1 (cols 0,1), B2 (cols 2,3), B3 (cols 4,5), B4 (cols 6,7)
    # Use a combined Strassen-like approach.
    #
    # Actually, let me use a known result: R(M_{2,2,8}) = 24
    # This can be achieved by: R = R(M_{2,2,4}) + R(M_{2,2,4}) with R(M_{2,2,4})=12
    # But 12+12=24.
    #
    # Let me implement rank-12 for M_{2,2,4} (which is safe and known).
    
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 24
    V = [[0] * rank for _ in range(m * p)]  # 16 x 24
    W = [[0] * rank for _ in range(p * n)]  # 16 x 24
    
    # For each 2x4 block, use a rank-12 decomposition
    # M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4
    # 
    # Rank-12 decomposition for M_{2,2,4}:
    # Use Strassen for the first 2x2 subblock (rank 7) and direct for the second (rank 5)?
    # No, that's 12. Let me use a systematic approach.
    #
    # Actually, the simplest safe approach: use Strassen for each 2x2x2 block.
    # 4 blocks * 7 = 28. That's what we have.
    #
    # Let me try to find if rank 24 is achievable with a different grouping.
    # 
    # I'll stick with the proven rank-28 Strassen approach but make sure it's correct.
    
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_coeffs = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    
    W_coeffs = [
        {0: 1, 3: 1},
        {1: 1, 3: -1},
        {2: 1, 3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {3: 1},
        {0: 1}
    ]

    for q in range(4):
        for s in range(7):
            r = q * 7 + s
            
            for u_idx, val in U_coeffs[s].items():
                U[u_idx][r] = val
                
            for v_local_idx, val in V_coeffs[s].items():
                j = v_local_idx // 2
                k_local = v_local_idx % 2
                k = 2 * q + k_local
                v_idx = j * p + k
                V[v_idx][r] = val
                
            for w_local_idx, val in W_coeffs[s].items():
                i = w_local_idx % 2
                k_local = w_local_idx // 2
                k = 2 * q + k_local
                w_idx = k * n + i
                W[w_idx][r] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
