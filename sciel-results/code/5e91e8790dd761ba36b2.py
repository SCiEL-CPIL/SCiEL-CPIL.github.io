"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # m1..m7
    # U coeffs for M[i, j]
    # V coeffs for N[i, j]
    # W coeffs for P[i, j]
    
    # Term 0: m1 = (M00+M11)(N00+N11)
    # M: (0,0):1, (1,1):1
    # N: (0,0):1, (1,1):1
    # P: (0,0):1, (1,1):1
    # Term 1: m2 = (M10+M11)N00
    # M: (1,0):1, (1,1):1
    # N: (0,0):1
    # P: (1,0):1, (1,1):-1
    # Term 2: m3 = M00(N10-N11)
    # M: (0,0):1
    # N: (1,0):1, (1,1):-1
    # P: (0,1):1, (1,0):1
    # Term 3: m4 = M11(N01-N00)
    # M: (1,1):1
    # N: (0,1):1, (0,0):-1
    # P: (0,0):-1, (0,1):1
    # Term 4: m5 = (M00+M01)N11
    # M: (0,0):1, (0,1):1
    # N: (1,1):1
    # P: (0,0):1, (1,1):1
    # Term 5: m6 = (M10-M00)(N00+N10)
    # M: (1,0):1, (0,0):-1
    # N: (0,0):1, (1,0):1
    # P: (1,1):1
    # Term 6: m7 = (M01-M11)(N01+N11)
    # M: (0,1):1, (1,1):-1
    # N: (0,1):1, (1,1):1
    # P: (0,0):1, (1,1):1
    
    strassen_U = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1), (0,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,1,1), (1,1,-1)]
    ]
    strassen_V = [
        [(0,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (0,0,-1)],
        [(1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,1,1), (1,1,1)]
    ]
    strassen_W = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(0,0,1), (1,1,1)],
        [(1,1,1)],
        [(0,0,1), (1,1,1)]
    ]
    
    for l in range(2): # A block index
        for k in range(4): # B block index
            s = k * 2 + l
            for c in range(7):
                col = s * 7 + c
                
                # Fill U
                for (i, j, val) in strassen_U[c]:
                    u_idx = i * 4 + (2 * l + j)
                    U[u_idx][col] += val
                    
                # Fill V
                for (i, j, val) in strassen_V[c]:
                    v_idx = (2 * l + i) * 8 + (2 * k + j)
                    V[v_idx][col] += val
                    
                # Fill W
                for (i, j, val) in strassen_W[c]:
                    w_idx = (2 * k + j) * 2 + i
                    W[w_idx][col] += val
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
