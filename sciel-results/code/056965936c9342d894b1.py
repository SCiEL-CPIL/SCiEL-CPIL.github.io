import time

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56 decomposition using Strassen for 8 sub-problems of 2x2x2
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen linear forms for X (2x2)
    # r=0: X00 + X11
    # r=1: X10 + X11
    # r=2: X00
    # r=3: X11
    # r=4: X00 + X01
    # r=5: X10 - X00
    # r=6: X01 - X11
    # Indices (i, j) for X
    U_forms = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    
    # Strassen linear forms for Y (2x2)
    # r=0: Y00 + Y11
    # r=1: Y00
    # r=2: Y01 - Y11
    # r=3: Y10 - Y00
    # r=4: Y11
    # r=5: Y00 + Y01
    # r=6: Y10 + Y11
    # Indices (k, l) for Y
    V_forms = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    
    # Reconstruction coefficients for Z (2x2)
    # Z00, Z01, Z10, Z11
    # r=0 (M1): Z00+1, Z11+1
    # r=1 (M2): Z10+1, Z11-1
    # r=2 (M3): Z01+1, Z11+1
    # r=3 (M4): Z00+1, Z10+1
    # r=4 (M5): Z00-1, Z01+1
    # r=5 (M6): Z11+1
    # r=6 (M7): Z00+1
    # Indices (i, l) for Z
    W_forms = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    for s in range(8):
        a = s // 4 # 0 or 1
        b = s % 4  # 0 to 3
        
        for r in range(7):
            c = s * 7 + r
            
            # Update U
            for i, j, coeff in U_forms[r]:
                # Global A index: row i, col a*2 + j
                u_idx = i * m + (a * 2 + j)
                U[u_idx][c] += coeff
                
            # Update V
            for k, l, coeff in V_forms[r]:
                # Global B index: row 2a + k, col 4b + l
                v_idx = (2 * a + k) * p + (4 * b + l)
                V[v_idx][c] += coeff
                
            # Update W
            for i, l, coeff in W_forms[r]:
                # Global C index: row i, col 4b + l
                # W index convention: W[k*n + i, r] where k is col, i is row
                # k_glob = 4b + l
                # i_glob = i
                w_idx = (4 * b + l) * n + i
                W[w_idx][c] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
