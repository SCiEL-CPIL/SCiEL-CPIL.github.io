import time

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds

    n, m, p = 2, 2, 10
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = A10(B00+B01)
    # M3 = (A00+A01)B11
    # M4 = A11(B10-B00)
    # M5 = (A00+A11)(B11-B10)
    # M6 = (A10+A11)B00
    # M7 = (A01-A11)(B10+B11)
    
    # U coefficients (A part)
    # U row indices: 0:A00, 1:A01, 2:A10, 3:A11
    U_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 0],  # M2
        [1, 1, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 0, 0, 1],  # M5
        [0, 0, 1, 1],  # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # V coefficients (B part)
    # V row indices for a block: 0:B00, 1:B01, 2:B10, 3:B11
    V_coeffs = [
        [1, 0, 1, 1],  # M1: B00+B11, M2: B00+B01, M3: B11, M4: B10-B00, M5: B11-B10, M6: B00, M7: B10+B11
        [0, 1, 0, 0],  # ...
        [0, 0, 1, 0],  # ...
        [0, 0, 0, 1]   # ...
    ]
    # Let's re-verify V_coeffs carefully.
    # M1: B00, B11 -> V[0]+=1, V[3]+=1
    # M2: B00, B01 -> V[0]+=1, V[1]+=1
    # M3: B11 -> V[3]+=1
    # M4: B10, B00 -> V[2]+=1, V[0]-=1
    # M5: B11, B10 -> V[3]+=1, V[2]-=1
    # M6: B00 -> V[0]+=1
    # M7: B10, B11 -> V[2]+=1, V[3]+=1
    
    V_coeffs = [
        [1, 0, 1, 1],  # M1: B00(1), B01(0), B10(1), B11(1) -- WAIT. M1 is (A00+A11)(B00+B11). So B00 and B11.
                       # Indices: 0:B00, 1:B01, 2:B10, 3:B11.
                       # So M1: [1, 0, 0, 1]
        [1, 1, 0, 0],  # M2: A10(B00+B01). So B00, B01. [1, 1, 0, 0]
        [0, 0, 0, 1],  # M3: (A00+A01)B11. So B11. [0, 0, 0, 1]
        [-1, 0, 1, 0], # M4: A11(B10-B00). So B10, B00(-1). [-1, 0, 1, 0]
        [-1, 0, -1, 1],# M5: (A00+A11)(B11-B10). So B11, B10(-1). [0, 0, -1, 1] -- WAIT. B00 is 0. So [0, 0, -1, 1]
        [1, 0, 0, 0],  # M6: (A10+A11)B00. So B00. [1, 0, 0, 0]
        [0, 0, 1, 1]   # M7: (A01-A11)(B10+B11). So B10, B11. [0, 0, 1, 1]
    ]
    
    # Corrected V_coeffs:
    V_coeffs = [
        [1, 0, 0, 1],  # M1
        [1, 1, 0, 0],  # M2
        [0, 0, 0, 1],  # M3
        [-1, 0, 1, 0], # M4
        [0, 0, -1, 1], # M5
        [1, 0, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    
    # W coefficients (C part)
    # W row indices for a block: 0:C00, 1:C01, 2:C10, 3:C11
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M4
    
    W_coeffs = [
        [1, 0, 0, 1],   # M1: C00(1), C01(0), C10(0), C11(1)
        [0, 0, 1, -1],  # M2: C00(0), C01(0), C10(1), C11(-1)
        [0, 1, 0, 1],   # M3: C00(0), C01(1), C10(0), C11(1)
        [1, 0, 0, 1],   # M4: C00(1), C01(0), C10(0), C11(1)
        [-1, 0, 1, 0],  # M5: C00(-1), C01(0), C10(1), C11(0)
        [0, 0, 0, 0],   # M6: C00(0), C01(0), C10(0), C11(0)
        [-1, 1, 0, 0]   # M7: C00(-1), C01(1), C10(0), C11(0)
    ]

    # Fill U
    for c in range(7):
        for i in range(4):
            U[i][c] = U_coeffs[c][i]
            
    # Fill V and W for each block
    num_blocks = p // 2  # 5 blocks
    for b in range(num_blocks):
        if time.time() > deadline - 10:
            break
        for c in range(7):
            col_idx = b * 7 + c
            # V
            for j in range(2):
                for k in range(2):
                    # B element index in V: j*p + (2*b + k)
                    v_row = j * p + (2 * b + k)
                    V[v_row][col_idx] += V_coeffs[c][j * 2 + k]
                    
            # W
            for i in range(2):
                for k in range(2):
                    # C element index in W: (2*b + k)*n + i
                    w_row = (2 * b + k) * n + i
                    W[w_row][col_idx] += W_coeffs[c][i * 2 + k]

    return {"U": U, "V": V, "W": W}

