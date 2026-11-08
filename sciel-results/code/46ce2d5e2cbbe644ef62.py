"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    # 8 independent 2x2x2 Strassen decompositions
    # q in {0,1}, t in {0,1,2,3}
    # Each Strassen product r in {0,...,6}
    
    rank = 8 * 7  # 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2x2 definitions
    # Each entry: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (i, s, coeff) meaning A[i, 2q+s] gets coeff
    # V_coeffs: list of (s, u, coeff) meaning B[2q+s, 2t+u] gets coeff
    # W_coeffs: list of (i, u, coeff) meaning C[i, 2t+u] gets coeff
    
    strassen = [
        # P1: (a+d)(e+h)
        ([(0,0,1),(1,1,1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,1,1)]),
        # P2: (c+d)(e)
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # P3: (a)(f-h)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # P4: (d)(g-e)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # P5: (a+b)(h)
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # P6: (c-a)(e+f)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # P7: (b-d)(g+h)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]
    
    col = 0
    for q in range(2):  # 2 q-blocks
        for t in range(4):  # 4 t-blocks
            for r in range(7):
                u_coeffs, v_coeffs, w_coeffs = strassen[r]
                for (i, s, coeff) in u_coeffs:
                    u_idx = i * m + (2 * q + s)
                    U[u_idx][col] += coeff
                for (s, u, coeff) in v_coeffs:
                    v_idx = (2 * q + s) * p + (2 * t + u)
                    V[v_idx][col] += coeff
                for (i, u, coeff) in w_coeffs:
                    w_idx = (2 * t + u) * n + i
                    W[w_idx][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
