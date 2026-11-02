"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=8
    # Use 4 independent Strassen 2x2x2 blocks, each of rank 7
    num_blocks = p // 2  # 4 blocks
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen 2x2x2 decomposition
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C = [[P1+P4-P5+P7, P3+P5], [P2+P4, P1-P2+P3+P6]]
    
    # U coefficients: each row is a coefficient vector for A entries [a, b, c, d]
    U_strassen = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1]   # P7: b-d
    ]
    
    # V coefficients: each row is a coefficient vector for B entries [e, f, g, h]
    V_strassen = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1]    # P7: g+h
    ]
    
    # W coefficients: each row is a coefficient vector for C entries [C00, C01, C10, C11]
    # C00 = P1+P4-P5+P7, C01 = P3+P5, C10 = P2+P4, C11 = P1-P2+P3+P6
    W_strassen = [
        [1, 0, 0, 1],   # P1: C00 + C11
        [0, 0, 1, -1],  # P2: C10 - C11
        [0, 1, 0, 1],   # P3: C01 + C11
        [1, 0, 1, 0],   # P4: C00 + C10
        [-1, 1, 0, 0],  # P5: -C00 + C01
        [0, 0, 0, 1],   # P6: C11
        [1, 0, 0, 0]    # P7: C00
    ]
    
    for b in range(num_blocks):
        base = b * 7
        # Local B column indices: 2b (e), 2b+1 (f), row 1: 8+2b (g), 8+2b+1 (h)
        # Local C indices: k*n+i where k=col, i=row
        # C[0][2b] -> k=2b, i=0 -> 2b*2+0 = 4b
        # C[0][2b+1] -> k=2b+1, i=0 -> (2b+1)*2+0 = 4b+2
        # C[1][2b] -> k=2b, i=1 -> 2b*2+1 = 4b+1
        # C[1][2b+1] -> k=2b+1, i=1 -> (2b+1)*2+1 = 4b+3
        
        v_local_to_global = [2*b, 2*b+1, 8+2*b, 8+2*b+1]
        w_local_to_global = [4*b, 4*b+2, 4*b+1, 4*b+3]
        
        for t in range(7):
            c = base + t
            
            # Fill U: row index = i*m+j for A[i][j]
            for i in range(n * m):
                U[i][c] = U_strassen[t][i]
            
            # Fill V
            for l in range(4):
                V[v_local_to_global[l]][c] = V_strassen[t][l]
            
            # Fill W
            for l in range(4):
                W[w_local_to_global[l]][c] = W_strassen[t][l]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
