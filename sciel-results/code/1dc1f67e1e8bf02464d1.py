import time

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses 7 independent Strassen blocks.
    """
    start_time = time.time()
    deadline = 120.0 # Safety margin
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: A00=0, A01=1, A10=2, A11=3
    U_forms = [
        [1, 0, 0, 1],   # M1 = A00 + A11
        [0, 0, 1, 1],   # M2 = A10 + A11
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = A00 + A01
        [-1, 0, 1, 0],  # M6 = A10 - A00
        [0, 1, 0, -1]   # M7 = A01 - A11
    ]
    
    for r in range(num_blocks):
        # B block indices
        # B is 2x14.
        # V shape is (m*p, rank) = (28, rank).
        # Row-major flattening of B (2x14):
        # B[j, k] is at index j*14 + k.
        
        idx_e = 2 * r       # B[0, 2r]
        idx_f = 2 * r + 1   # B[0, 2r+1]
        idx_g = 14 + 2 * r       # B[1, 2r]
        idx_h = 14 + 2 * r + 1   # B[1, 2r+1]
        
        # W shape is (p*n, rank) = (28, rank).
        # Convention: W[k*n + i, r].
        # C is 2x14. C[i, k].
        # Index in W: k*n + i = 2*k + i.
        
        w_idx_c00 = 4 * r       # C[0, 2r] -> 2*(2r) + 0
        w_idx_c10 = 4 * r + 1   # C[1, 2r] -> 2*(2r) + 1
        w_idx_c01 = 4 * r + 2   # C[0, 2r+1] -> 2*(2r+1) + 0
        w_idx_c11 = 4 * r + 3   # C[1, 2r+1] -> 2*(2r+1) + 1
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
                
            # Fill V
            if s == 0:
                V[idx_e][c] = 1.0
                V[idx_h][c] = 1.0
            elif s == 1:
                V[idx_e][c] = 1.0
            elif s == 2:
                V[idx_f][c] = 1.0
                V[idx_h][c] = -1.0
            elif s == 3:
                V[idx_e][c] = -1.0
                V[idx_g][c] = 1.0
            elif s == 4:
                V[idx_h][c] = 1.0
            elif s == 5:
                V[idx_e][c] = 1.0
                V[idx_f][c] = 1.0
            elif s == 6:
                V[idx_g][c] = 1.0
                V[idx_h][c] = 1.0
                
            # Fill W
            if s == 0:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 1:
                W[w_idx_c10][c] = 1.0
                W[w_idx_c11][c] = -1.0
            elif s == 2:
                W[w_idx_c01][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 3:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c10][c] = 1.0
            elif s == 4:
                W[w_idx_c00][c] = -1.0
                W[w_idx_c01][c] = 1.0
            elif s == 5:
                W[w_idx_c11][c] = 1.0
            elif s == 6:
                W[w_idx_c00][c] = 1.0
                
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
