import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses hierarchical Strassen decomposition for 4x4x4 (rank 49) applied twice.
    Total rank: 98.
    """
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Define rank-7 Strassen factors for 2x2x2
    # U2, V2, W2 shapes (4, 7)
    # Indices: U: i*2+j, V: j*2+k, W: k*2+i
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 0, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [0, 1, 0, 1, 1, 0, -1]   # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 1, 1, 0],  # B00
        [0, 0, 1, 0, 0, 0, 0],   # B01
        [0, 0, 0, 1, 1, 0, 1],   # B10
        [1, 0, -1, 0, 0, 1, 1]   # B11
    ]
    
    W2 = [
        [1, 0, 0, -1, 0, -1, 1],  # C00 (k=0, i=0)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=0, i=1)
        [0, 0, 1, 0, 0, 0, 1],    # C01 (k=1, i=0)
        [1, -1, -1, 0, 1, 0, 0]   # C11 (k=1, i=1)
    ]
    
    # Construct rank-49 factors for 4x4x4
    # U4, V4, W4 shapes (16, 49)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r in range(7): # Block level rank
        for s in range(7): # Element level rank
            R = r * 7 + s
            
            for p in range(2): # Block row of A
                for q in range(2): # Block col of A
                    coeff_U_B = U2[p*2+q][r]
                    if coeff_U_B == 0: continue
                    
                    for i_el in range(2):
                        for j_el in range(2):
                            coeff_U_2 = U2[i_el*2+j_el][s]
                            if coeff_U_2 == 0: continue
                            
                            i = p*2 + i_el
                            j = q*2 + j_el
                            U4[i*4+j][R] += coeff_U_B * coeff_U_2

            for q in range(2): # Block row of B (col of A)
                for s_b in range(2): # Block col of B
                    coeff_V_B = V2[q*2+s_b][r]
                    if coeff_V_B == 0: continue
                    
                    for j_el in range(2):
                        for k_el in range(2):
                            coeff_V_2 = V2[j_el*2+k_el][s]
                            if coeff_V_2 == 0: continue
                            
                            j = q*2 + j_el
                            k = s_b*2 + k_el
                            V4[j*4+k][R] += coeff_V_B * coeff_V_2

            for s_b in range(2): # Block row of C (col of B)
                for p in range(2): # Block col of C (row of A)
                    coeff_W_B = W2[s_b*2+p][r]
                    if coeff_W_B == 0: continue
                    
                    for k_el in range(2):
                        for i_el in range(2):
                            coeff_W_2 = W2[k_el*2+i_el][s]
                            if coeff_W_2 == 0: continue
                            
                            k = s_b*2 + k_el
                            i = p*2 + i_el
                            W4[k*4+i][R] += coeff_W_B * coeff_W_2

    # Construct rank-98 factors for 4x4x8
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First 49 ranks: k in 0..3
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                V[j*8+k][r] = V4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[k*4+i][r] = W4[k*4+i][r]
                
    # Last 49 ranks: k in 4..7
    for r in range(49):
        for i in range(16):
            U[i][r+49] = U4[i][r]
        for j in range(4):
            for k in range(4):
                V[j*8+(k+4)][r+49] = V4[j*4+k][r]
        for k in range(4):
            for i in range(4):
                W[(k+4)*4+i][r+49] = W4[k*4+i][r]
    
    return {"U": U, "V": V, "W": W}
