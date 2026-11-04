"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 24 decomposition for 2x4x4 matrix multiplication
    # Using a known rank-24 scheme based on 3x2x2 Strassen-like blocks
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 8 x 24
    V = [[0] * rank for _ in range(m * p)]  # 16 x 24
    W = [[0] * rank for _ in range(p * n)]  # 8 x 24

    # We use 4 independent 2x2x2 Strassen blocks, but share terms across blocks
    # Actually, let me use a simpler approach: 4 Strassen blocks = rank 28
    # But we can do better by using the fact that A is only 2 rows.
    
    # Let me try a direct rank-24 construction.
    # The 2x4x4 tensor has 2*4*4 = 32 entries.
    # Rank 24 means we need 24 multiplications.
    
    # One approach: use 6 Strassen 2x2x2 blocks (each rank 7) = rank 42, too high.
    # Use 4 blocks = rank 28.
    
    # Let me just implement the rank 28 solution correctly first, then see if I can reduce.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    strassen_terms = [
        (
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1},
            {(0,0): 1, (1,1): 1}
        ),
        (
            {(1,0): 1, (1,1): 1},
            {(0,0): 1},
            {(1,0): 1, (1,1): -1}
        ),
        (
            {(0,0): 1},
            {(0,1): 1, (1,1): -1},
            {(0,1): 1, (1,1): 1}
        ),
        (
            {(1,1): 1},
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (1,0): 1}
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(1,1): 1},
            {(0,0): -1, (0,1): 1}
        ),
        (
            {(1,0): 1, (0,0): -1},
            {(0,0): 1, (0,1): 1},
            {(1,1): 1}
        ),
        (
            {(0,1): 1, (1,1): -1},
            {(1,0): 1, (1,1): 1},
            {(0,0): 1}
        )
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for b_idx in range(4):
        a_ro, a_co, b_ro, b_co, c_ro, c_co = blocks[b_idx]
        
        for t_idx in range(7):
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            
            for (li, lj), coeff in u_coeffs.items():
                gi = a_ro + li
                gj = a_co + lj
                U[gi * m + gj][col] += coeff
            
            for (lj, lk), coeff in v_coeffs.items():
                gj = b_ro + lj
                gk = b_co + lk
                V[gj * p + gk][col] += coeff
            
            for (li, lk), coeff in w_coeffs.items():
                gi = c_ro + li
                gk = c_co + lk
                W[gk * n + gi][col] += coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
