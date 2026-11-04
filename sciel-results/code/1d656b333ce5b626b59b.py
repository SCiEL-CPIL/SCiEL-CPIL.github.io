"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for x in range(2):
        for y in range(4):
            # Strassen steps for 2x2x2
            # x_coeffs, y_coeffs, z_coeffs
            steps = [
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
                    {(1,0): 1, (1,1): -1},
                    {(0,1): 1, (1,1): 1}
                ),
                (
                    {(1,1): 1},
                    {(0,1): 1, (0,0): -1},
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
                ),
            ]
            
            for x_c, y_c, z_c in steps:
                for (r, c), val in x_c.items():
                    u_idx = r * m + (c + 2 * x)
                    U[u_idx][col] += val
                
                for (r, c), val in y_c.items():
                    v_idx = (r + 2 * x) * p + (c + 2 * y)
                    V[v_idx][col] += val
                
                for (r, c), val in z_c.items():
                    w_idx = (2 * y + c) * n + r
                    W[w_idx][col] += val
                
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
