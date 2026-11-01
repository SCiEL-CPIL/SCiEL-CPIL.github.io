"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 (Strassen rank) = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U_coeffs[s][i*2+j]
    # V_coeffs[s][j*2+r] (local B index)
    # W_coeffs[s][r*2+i] (local C index, note: C is n x 2, index r is col, i is row)
    # Local B index l: j = l//2, r = l%2
    # Local C index l: i = l%2, r = l//2  (Wait, C_{i,r}. l=0->C00, l=1->C01, l=2->C10, l=3->C11)
    # So i = l % 2, r = l // 2.
    
    U_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    V_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_coeffs = [
        [1, 0, 0, 1], # M1 -> C00, C11
        [0, 0, 1, -1],# M2 -> C10, -C11
        [0, 1, 0, 1], # M3 -> C01, C11
        [1, 0, 1, 0], # M4 -> C00, C10
        [-1, 1, 0, 0],# M5 -> -C00, C01
        [0, 0, 0, 1], # M6 -> C11
        [1, 0, 0, 0]  # M7 -> C00
    ]
    
    # 4 blocks
    for q in range(4):
        for s in range(7):
            c = q * 7 + s
            
            # Fill U
            for idx in range(4):
                val = U_coeffs[s][idx]
                if val != 0:
                    U[idx][c] += val
                    
            # Fill V
            for l in range(4):
                val = V_coeffs[s][l]
                if val != 0:
                    j = l // 2
                    r = l % 2
                    k = 2 * q + r
                    v_row = j * p + k
                    V[v_row][c] += val
                    
            # Fill W
            for l in range(4):
                val = W_coeffs[s][l]
                if val != 0:
                    i = l % 2
                    r = l // 2
                    k = 2 * q + r
                    w_row = k * n + i
                    W[w_row][c] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
