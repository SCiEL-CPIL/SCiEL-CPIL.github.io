"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # Each product is a tuple (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: list of (i, j, coeff)
    # v_coeffs: list of (j, k, coeff)
    # w_coeffs: list of (k, i, coeff)
    
    strassen_products = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # M6
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    col = 0
    # 4 blocks: (j_offset, k_offset)
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for u_coeffs, v_coeffs, w_coeffs in strassen_products:
                # Fill U
                for i, j, c in u_coeffs:
                    row_u = i * m + (j_off + j)
                    U[row_u][col] += c
                # Fill V
                for j, k, c in v_coeffs:
                    row_v = (j_off + j) * p + (k_off + k)
                    V[row_v][col] += c
                # Fill W
                for k, i, c in w_coeffs:
                    row_w = (k_off + k) * n + i
                    W[row_w][col] += c
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
