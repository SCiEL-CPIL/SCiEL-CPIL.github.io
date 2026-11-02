"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for M_2,2,2
    # M1 = (x00 + x11)(y00 + y11)
    # M2 = (x10 + x11)y00
    # M3 = x00(y01 - y11)
    # M4 = x11(y10 - y00)
    # M5 = (x00 + x01)y11
    # M6 = (x10 - x00)(y00 + y01)
    # M7 = (x01 - x11)(y10 + y11)
    
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    sub_problems = [
        (0, 0, 0, 0),
        (0, 2, 2, 0),
        (0, 0, 0, 2),
        (0, 2, 2, 2),
    ]
    
    col = 0
    for (i_off, j_off_A, j_off_B, k_off) in sub_problems:
        for x_coeffs, y_coeffs, z_coeffs in strassen_terms:
            for (xi, xj), coeff in x_coeffs.items():
                u_idx = (i_off + xi) * m + (j_off_A + xj)
                U[u_idx][col] += coeff
                
            for (yi, yj), coeff in y_coeffs.items():
                v_idx = (j_off_B + yi) * p + (k_off + yj)
                V[v_idx][col] += coeff
                
            for (zi, zj), coeff in z_coeffs.items():
                w_idx = (k_off + zj) * n + (i_off + zi)
                W[w_idx][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
