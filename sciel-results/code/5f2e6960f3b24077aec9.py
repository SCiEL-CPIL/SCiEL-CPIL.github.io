import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    
    The problem decomposes into 7 independent 2x2 * 2x2 matrix multiplications.
    Each 2x2 * 2x2 multiplication is solved using Strassen's algorithm (rank 7).
    Total rank = 7 * 7 = 49.
    
    This is the theoretical minimum rank for this tensor, as the tensor is a 
    direct sum of 7 copies of the 2x2x2 matrix multiplication tensor, and 
    the rank of the 2x2x2 tensor is 7.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
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
    
    # Pre-compute V and W coefficients for each Strassen step s
    # V coefficients depend on block r
    # W coefficients depend on block r
    
    for r in range(num_blocks):
        # B block indices
        # B is 2x14. Index j*p + k.
        idx_e = 2 * r       # B[0, 2r]
        idx_f = 2 * r + 1   # B[0, 2r+1]
        idx_g = 14 + 2 * r       # B[1, 2r]
        idx_h = 14 + 2 * r + 1   # B[1, 2r+1]
        
        # W indices for C block r
        # C is 14x2 (conceptually, but stored as p*n). Index k*n + i.
        w_idx_c00 = 4 * r       # C[0, 2r]
        w_idx_c10 = 4 * r + 1   # C[1, 2r]
        w_idx_c01 = 4 * r + 2   # C[0, 2r+1]
        w_idx_c11 = 4 * r + 3   # C[1, 2r+1]
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = float(U_forms[s][i])
                
            # Fill V
            # s=0 (M1): e+h
            # s=1 (M2): e
            # s=2 (M3): f-h
            # s=3 (M4): g-e
            # s=4 (M5): h
            # s=5 (M6): e+f
            # s=6 (M7): g+h
            
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
            # s=0 (M1): C00+1, C11+1
            # s=1 (M2): C10+1, C11-1
            # s=2 (M3): C01+1, C11+1
            # s=3 (M4): C00+1, C10+1
            # s=4 (M5): C00-1, C01+1
            # s=5 (M6): C11+1
            # s=6 (M7): C00+1
            
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
# EVOLVE-BLOCK-END
