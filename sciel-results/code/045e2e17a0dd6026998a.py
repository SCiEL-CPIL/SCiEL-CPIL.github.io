import time
import random

DIMENSIONS = [2, 2, 14]

def construct():
    """
    Rank-49 decomposition for 2x2 * 2x14 matrix multiplication.
    Uses 7 independent Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    U_forms = [
        [1, 0, 0, 1],   # M1 = A00 + A11
        [0, 0, 1, 1],   # M2 = A10 + A11
        [1, 0, 0, 0],   # M3 = A00
        [0, 0, 0, 1],   # M4 = A11
        [1, 1, 0, 0],   # M5 = A00 + A01
        [-1, 0, 1, 0],  # M6 = A10 - A00
        [0, 1, 0, -1]   # M7 = A01 - A11
    ]
    
    # Strassen B-forms for a 2x2 block [e, f; g, h]
    # Local indices: 0=e(B[0,0]), 1=f(B[0,1]), 2=g(B[1,0]), 3=h(B[1,1])
    V_forms = [
        [1, 0, 0, 1],   # M1: e + h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f - h
        [-1, 0, 1, 0],  # M4: g - e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e + f
        [0, 0, 1, 1]    # M7: g + h
    ]
    
    # Strassen C-combinations
    # Local indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
    W_forms = [
        [1, 0, 0, 1],   # M1: C00 + 1, C11 + 1
        [0, 0, 1, -1],  # M2: C10 + 1, C11 - 1
        [0, 1, 0, 1],   # M3: C01 + 1, C11 + 1
        [1, 0, 1, 0],   # M4: C00 + 1, C10 + 1
        [-1, 1, 0, 0],  # M5: C00 - 1, C01 + 1
        [0, 0, 0, 1],   # M6: C11 + 1
        [1, 0, 0, 0]    # M7: C00 + 1
    ]
    
    for r in range(num_blocks):
        # B block indices in V (shape m*p = 2*14 = 28)
        # B is 2x14. Row 0: 0..13, Row 1: 14..27.
        # Block r uses columns 2r, 2r+1.
        
        v_idx_e = 2 * r
        v_idx_f = 2 * r + 1
        v_idx_g = 14 + 2 * r
        v_idx_h = 14 + 2 * r + 1
        
        # W indices for C in block r
        # W shape (p*n = 14*2 = 28). Convention W[k*n + i, r].
        
        w_idx_c00 = 4 * r
        w_idx_c01 = 4 * r + 2
        w_idx_c10 = 4 * r + 1
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            V[v_idx_e][c] = V_forms[s][0]
            V[v_idx_f][c] = V_forms[s][1]
            V[v_idx_g][c] = V_forms[s][2]
            V[v_idx_h][c] = V_forms[s][3]
            
            # Fill W
            W[w_idx_c00][c] = W_forms[s][0]
            W[w_idx_c01][c] = W_forms[s][1]
            W[w_idx_c10][c] = W_forms[s][2]
            W[w_idx_c11][c] = W_forms[s][3]
    
    return {"U": U, "V": V, "W": W}
