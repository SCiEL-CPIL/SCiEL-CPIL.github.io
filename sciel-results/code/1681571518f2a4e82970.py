import time

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen definitions for 2x2
    # A indices: 00, 01, 10, 11 -> U rows: 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> V rows: 8*0+0, 8*0+1, 8*1+0, 8*1+1 -> 0, 1, 8, 9
    # C indices: 00, 01, 10, 11 -> W rows: 2*0+0, 2*1+0, 2*0+1, 2*1+1 -> 0, 2, 1, 3
    
    # We define the 7 steps for a single block b
    # Each step is a tuple (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {row: val}
    # V_coeffs: dict {row: val}
    # W_coeffs: dict {row: val}
    
    # Block b uses B columns 2b and 2b+1
    # V rows for B_00, B_01, B_10, B_11 in block b:
    # B_00 -> j=0, k=2b -> row 2b
    # B_01 -> j=0, k=2b+1 -> row 2b+1
    # B_10 -> j=1, k=2b -> row 8+2b
    # B_11 -> j=1, k=2b+1 -> row 8+2b+1
    
    steps = []
    
    # Step 1: M1 = (A00+A11)(B00+B11)
    # C00 += M1, C11 += M1
    steps.append({
        'U': {0: 1, 3: 1},
        'V': lambda b: {2*b: 1, 8+2*b+1: 1},
        'W': {0: 1, 3: 1}
    })
    
    # Step 2: M2 = (A10+A11)B00
    # C10 += M2, C11 += -M2
    steps.append({
        'U': {2: 1, 3: 1},
        'V': lambda b: {2*b: 1},
        'W': {1: 1, 3: -1}
    })
    
    # Step 3: M3 = A00(B01-B11)
    # C01 += M3, C11 += M3
    steps.append({
        'U': {0: 1},
        'V': lambda b: {2*b+1: 1, 8+2*b+1: -1},
        'W': {2: 1, 3: 1}
    })
    
    # Step 4: M4 = A11(B10-B00)
    # C00 += M4, C10 += M4
    steps.append({
        'U': {3: 1},
        'V': lambda b: {8+2*b: 1, 2*b: -1},
        'W': {0: 1, 1: 1}
    })
    
    # Step 5: M5 = (A00+A01)B11
    # C00 += -M5, C01 += M5
    steps.append({
        'U': {0: 1, 1: 1},
        'V': lambda b: {8+2*b+1: 1},
        'W': {0: -1, 2: 1}
    })
    
    # Step 6: M6 = (A10-A00)(B00+B01)
    # C11 += M6
    steps.append({
        'U': {2: 1, 0: -1},
        'V': lambda b: {2*b: 1, 2*b+1: 1},
        'W': {3: 1}
    })
    
    # Step 7: M7 = (A01-A11)(B10+B11)
    # C00 += M7
    steps.append({
        'U': {1: 1, 3: -1},
        'V': lambda b: {8+2*b: 1, 8+2*b+1: 1},
        'W': {0: 1}
    })
    
    for b in range(4):
        for s_idx in range(7):
            r = 7 * b + s_idx
            step = steps[s_idx]
            for u_row, val in step['U'].items():
                U[u_row][r] = val
            for v_row, val in step['V'](b).items():
                V[v_row][r] = val
            for w_row, val in step['W'].items():
                W[w_row][r] = val
                
    return {"U": U, "V": V, "W": W}
