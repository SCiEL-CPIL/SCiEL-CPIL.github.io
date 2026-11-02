import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # m=4, p=6
    # We split B into 3 blocks of 4x2.
    # Each block uses a rank-14 decomposition for 2x4 by 4x2.
    
    # Define the rank-14 decomposition for 2x4 by 4x2.
    # Local dimensions: n=2, m=4, p=2.
    # U: 8x14, V: 8x14, W: 4x14
    
    # Strassen terms for 2x2 by 2x2.
    # x indices: (i, j) with i in {0,1}, j in {0,1}
    # y indices: (j, k) with j in {0,1}, k in {0,1}
    # z indices: (i, k) with i in {0,1}, k in {0,1}
    
    # We define the 7 Strassen multiplications.
    # Each term t has U_coeffs, V_coeffs, W_coeffs.
    # U_coeffs: dict mapping (i, j) to val
    # V_coeffs: dict mapping (j, k) to val
    # W_coeffs: dict mapping (k, i) to val
    
    strassen_terms = []
    
    # M1 = (x00 + x11)(y00 + y11)
    strassen_terms.append({
        'U': {(0,0): 1, (1,1): 1},
        'V': {(0,0): 1, (1,0): 1, (0,1): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # M2 = (x10 + x11)y00
    strassen_terms.append({
        'U': {(1,0): 1, (1,1): 1},
        'V': {(0,0): 1},
        'W': {(1,0): 1, (1,1): 1}
    })
    
    # M3 = x00(y01 - y11)
    strassen_terms.append({
        'U': {(0,0): 1},
        'V': {(0,1): 1, (1,1): -1},
        'W': {(0,1): 1, (1,1): 1}
    })
    
    # M4 = x11(y10 - y00)
    strassen_terms.append({
        'U': {(1,1): 1},
        'V': {(1,0): 1, (0,0): -1},
        'W': {(1,0): 1, (1,1): 1}
    })
    
    # M5 = (x00 + x01)y11
    strassen_terms.append({
        'U': {(0,0): 1, (0,1): 1},
        'V': {(1,1): 1},
        'W': {(0,0): -1, (0,1): 1}
    })
    
    # M6 = (x10 - x00)(y00 + y01)
    strassen_terms.append({
        'U': {(1,0): 1, (0,0): -1},
        'V': {(0,0): 1, (0,1): 1},
        'W': {(1,1): 1}
    })
    
    # M7 = (x01 - x11)(y10 + y11)
    strassen_terms.append({
        'U': {(0,1): 1, (1,1): -1},
        'V': {(1,0): 1, (1,1): 1},
        'W': {(0,0): 1, (1,1): 1}
    })
    
    # Construct U_local, V_local, W_local for 2x4 by 4x2.
    # U_local: 8x14. Rows: i*4+j. i in {0,1}, j in {0,1,2,3}.
    # V_local: 8x14. Rows: j*2+k. j in {0,1,2,3}, k in {0,1}.
    # W_local: 4x14. Rows: k*2+i. k in {0,1}, i in {0,1}.
    
    U_local = [[0]*14 for _ in range(8)]
    V_local = [[0]*14 for _ in range(8)]
    W_local = [[0]*14 for _ in range(4)]
    
    # Block 1: A1 (j=0,1) and B1 (j=0,1). Terms 0-6.
    for t_idx in range(7):
        term = strassen_terms[t_idx]
        # Map local (i, j) to global U row i*4+j
        for (i, j), val in term['U'].items():
            U_local[i*4 + j][t_idx] = val
        # Map local (j, k) to global V row j*2+k
        for (j, k), val in term['V'].items():
            V_local[j*2 + k][t_idx] = val
        # Map local (k, i) to global W row k*2+i
        for (k, i), val in term['W'].items():
            W_local[k*2 + i][t_idx] = val
            
    # Block 2: A2 (j=2,3) and B2 (j=2,3). Terms 7-13.
    for t_idx in range(7):
        term = strassen_terms[t_idx]
        global_t = 7 + t_idx
        # Map local (i, j') to global U row i*4 + (j'+2)
        for (i, j), val in term['U'].items():
            U_local[i*4 + (j+2)][global_t] = val
        # Map local (j', k) to global V row (j'+2)*2 + k
        for (j, k), val in term['V'].items():
            V_local[(j+2)*2 + k][global_t] = val
        # Map local (k, i) to global W row k*2+i
        for (k, i), val in term['W'].items():
            W_local[k*2 + i][global_t] = val

    # Now construct full U, V, W for 2x4 by 4x6.
    # Rank = 3 * 14 = 42.
    rank = 42
    U = [[0]*rank for _ in range(8)]
    V = [[0]*rank for _ in range(24)] # 4*6
    W = [[0]*rank for _ in range(12)] # 6*2
    
    # 3 blocks of B, each 4x2.
    for block_idx in range(3):
        # U is the same for all blocks (A is shared).
        for r in range(8):
            for c in range(14):
                U[r][block_idx*14 + c] = U_local[r][c]
                
        # V: local row j*2+k maps to global row j*6 + (2*block_idx + k)
        for r_local in range(8):
            # r_local = j*2 + k
            j = r_local // 2
            k = r_local % 2
            r_global = j * 6 + (2 * block_idx + k)
            for c in range(14):
                V[r_global][block_idx*14 + c] = V_local[r_local][c]
                
        # W: local row k*2+i maps to global row (2*block_idx + k)*2 + i
        for r_local in range(4):
            # r_local = k*2 + i
            k = r_local // 2
            i = r_local % 2
            r_global = (2 * block_idx + k) * 2 + i
            for c in range(14):
                W[r_global][block_idx*14 + c] = W_local[r_local][c]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
