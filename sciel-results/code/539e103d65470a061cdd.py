import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses Strassen's algorithm for each of the 7 independent 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7  # 49
    
    # Initialize U, V, W as lists of lists to match interface, 
    # but we can build with numpy and convert or just use lists.
    # The interface expects lists of lists of floats/ints.
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A flat indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    U_forms = np.array([
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ])
    
    for r in range(num_blocks):
        # B indices for block r (cols 2r, 2r+1)
        # V flat index: j*p + k
        idx_e = 2 * r       # B[0, 2r]
        idx_f = 2 * r + 1   # B[0, 2r+1]
        idx_g = 14 + 2 * r       # B[1, 2r]
        idx_h = 14 + 2 * r + 1   # B[1, 2r+1]
        
        # W indices for C block r (rows 2r, 2r+1)
        # W flat index: k*n + i
        w_idx_c00 = 4 * r       # C[2r, 0]
        w_idx_c10 = 4 * r + 1   # C[2r, 1]
        w_idx_c01 = 4 * r + 2   # C[2r+1, 0]
        w_idx_c11 = 4 * r + 3   # C[2r+1, 1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
                
            # Fill V based on Strassen M definitions
            # M1: (A00+A11)(B00+B11) -> V uses B00, B11
            # Note: In our mapping, B00 is idx_e, B11 is idx_h
            # B01 is idx_f, B10 is idx_g
            
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
                
    return {"U": U, "V": V, "W": W}
