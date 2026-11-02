"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 decomposition using Strassen's algorithm on 2x2x2 subproblems.
    A (2x4) * B (4x8) = C (2x8)
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen's 7 factors for 2x2 * 2x2
    strassen_factors = [
        ([1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 1, 0], [1, 0, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1]),
        ([0, 1, 0, 0], [0, 0, 0, 1]),
        ([1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1]),
        ([1, 0, -1, 0], [1, 1, 0, 0]),
    ]
    
    # Output reconstruction: Z[i,j] from M1..M7
    output_coeffs = [
        [-1, 0, 1, 0, 1, 0, -1],  # Z00
        [1, 0, 0, 1, 0, 0, 0],    # Z01
        [0, 1, 1, 0, 0, 0, 0],    # Z10
        [0, -1, 0, 1, 1, -1, 0],  # Z11
    ]
    
    for j in range(4):
        # A_L * B_Tj: 7 factors
        for s in range(7):
            x_coeffs, y_coeffs = strassen_factors[s]
            u_idx = [0, 1, 2, 3]
            v_idx = [4 * j, 4 * j + 1, 4 * j + 2, 4 * j + 3]
            
            for i, c in enumerate(x_coeffs):
                if c != 0:
                    U[u_idx[i]][col] += c
            for i, c in enumerate(y_coeffs):
                if c != 0:
                    V[v_idx[i]][col] += c
            
            w_idx = [4 * j, 4 * j + 2, 4 * j + 1, 4 * j + 3]
            for i in range(4):
                W[w_idx[i]][col] += output_coeffs[i][s]
            
            col += 1
        
        # A_R * B_Bj: 7 factors
        for s in range(7):
            x_coeffs, y_coeffs = strassen_factors[s]
            u_idx = [4, 5, 6, 7]
            v_idx = [4 * j + 4, 4 * j + 5, 4 * j + 6, 4 * j + 7]
            
            for i, c in enumerate(x_coeffs):
                if c != 0:
                    U[u_idx[i]][col] += c
            for i, c in enumerate(y_coeffs):
                if c != 0:
                    V[v_idx[i]][col] += c
            
            w_idx = [4 * j, 4 * j + 2, 4 * j + 1, 4 * j + 3]
            for i in range(4):
                W[w_idx[i]][col] += output_coeffs[i][s]
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
