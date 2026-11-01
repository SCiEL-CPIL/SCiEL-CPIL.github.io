"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition using Strassen's algorithm on 2x2 blocks of the second matrix.
    
    The tensor is decomposed into 7 blocks, each corresponding to a 2x2 by 2x2 multiplication.
    Each block uses the 7-term Strassen decomposition.
    Total rank = 7 * 7 = 49.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: A00=0, A01=1, A10=2, A11=3
    U_forms = [
        [1, 0, 0, 1],   # F1 = A00+A11
        [0, 0, 1, 1],   # F2 = A10+A11
        [1, 0, 0, 0],   # F3 = A00
        [0, 0, 0, 1],   # F4 = A11
        [1, 1, 0, 0],   # F5 = A00+A01
        [-1, 0, 1, 0],  # F6 = A10-A00
        [0, 1, 0, -1]   # F7 = A01-A11
    ]
    
    for r in range(num_blocks):
        if time.time() > deadline:
            break
        
        # Block r uses B columns 2r and 2r+1
        idx_e = 2 * r          # B[0, 2r]
        idx_f = 2 * r + 1      # B[0, 2r+1]
        idx_g = 14 + 2 * r     # B[1, 2r]
        idx_h = 15 + 2 * r     # B[1, 2r+1]
        
        # W indices for C in block r: C[i, 2r+s] -> W[(2r+s)*n + i]
        w_idx_c00 = (2 * r) * n + 0      # C[0, 2r]
        w_idx_c10 = (2 * r) * n + 1      # C[1, 2r]
        w_idx_c01 = (2 * r + 1) * n + 0  # C[0, 2r+1]
        w_idx_c11 = (2 * r + 1) * n + 1  # C[1, 2r+1]
        
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
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
