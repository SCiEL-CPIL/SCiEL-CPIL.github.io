"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 7 blocks of 2 cols (rank 7 each) + 1 block of 1 col (rank 4)
    # 7 * 7 + 4 = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # u: [A00, A01, A10, A11]
    # v: [B00, B01, B10, B11] (local cols 0, 1)
    # w: [C00, C01, C10, C11] (local cols 0, 1)
    
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # C00=1, C11=1
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # C10=1, C11=-1
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # C01=1, C11=1
        # M4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]), # C00=1, C10=1
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # C00=-1, C01=1
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # C11=1
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # C00=1
    ]
    
    # Schoolbook for 2x2x1
    # u: [A00, A01, A10, A11]
    # v: [B0, B1] (local col 0)
    # w: [C0, C1] (local col 0)
    
    schoolbook_terms_1x1 = [
        ([1, 0, 0, 0], [1, 0], [1, 0]), # A00 B0 -> C0
        ([0, 1, 0, 0], [0, 1], [1, 0]), # A01 B1 -> C0
        ([0, 0, 1, 0], [1, 0], [0, 1]), # A10 B0 -> C1
        ([0, 0, 0, 1], [0, 1], [0, 1]), # A11 B1 -> C1
    ]
    
    col = 0
    
    # Process 7 blocks of 2 columns
    for g in range(7):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        for u, v, w in strassen_terms:
            # U
            for idx_a, val in enumerate(u):
                if val != 0:
                    i = idx_a // 2
                    j = idx_a % 2
                    U[i * m + j][col] += val
            
            # V
            # v is [B00, B01, B10, B11]
            # Local B indices: (0,0), (0,1), (1,0), (1,1)
            # Global B indices: (0, k0), (0, k1), (1, k0), (1, k1)
            for idx_b, val in enumerate(v):
                if val != 0:
                    j = idx_b // 2
                    l = idx_b % 2
                    k = k0 if l == 0 else k1
                    V[j * p + k][col] += val
            
            # W
            # w is [C00, C01, C10, C11]
            # Local C indices: (0,0), (0,1), (1,0), (1,1)
            # Global C indices: (0, k0), (0, k1), (1, k0), (1, k1)
            # W index: k * n + i
            for idx_c, val in enumerate(w):
                if val != 0:
                    i = idx_c // 2
                    l = idx_c % 2
                    k = k0 if l == 0 else k1
                    W[k * n + i][col] += val
            
            col += 1
    
    # Process 1 block of 1 column (k=14)
    k0 = 14
    
    for u, v, w in schoolbook_terms_1x1:
        # U
        for idx_a, val in enumerate(u):
            if val != 0:
                i = idx_a // 2
                j = idx_a % 2
                U[i * m + j][col] += val
        
        # V
        # v is [B0, B1]
        # Local B indices: (0,0), (1,0)
        # Global B indices: (0, k0), (1, k0)
        for idx_b, val in enumerate(v):
            if val != 0:
                j = idx_b
                k = k0
                V[j * p + k][col] += val
        
        # W
        # w is [C0, C1]
        # Local C indices: (0,0), (1,0)
        # Global C indices: (0, k0), (1, k0)
        # W index: k * n + i
        for idx_c, val in enumerate(w):
            if val != 0:
                i = idx_c
                k = k0
                W[k * n + i][col] += val
        
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
