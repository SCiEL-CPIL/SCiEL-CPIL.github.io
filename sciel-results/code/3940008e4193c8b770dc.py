import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS
    rank = 88

    u_rows = n * m  # 12
    v_rows = m * p  # 32
    w_rows = p * n  # 24

    U_np = np.zeros((u_rows, rank), dtype=float)
    V_np = np.zeros((v_rows, rank), dtype=float)
    W_np = np.zeros((w_rows, rank), dtype=float)

    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(m):
        for k in range(p):
            if time.time() > deadline:
                return {"U": [list(r) for r in U_np], "V": [list(r) for r in V_np], "W": [list(r) for r in W_np]}
            U_np[(0 * m + j), col] = 1
            V_np[(j * p + k), col] = 1
            W_np[(k * n + 0), col] = 1
            col += 1

    if time.time() > deadline:
        return {"U": [list(r) for r in U_np], "V": [list(r) for r in V_np], "W": [list(r) for r in W_np]}

    # Part 2: i=1,2 using Strassen 2x2x2 blocks
    # Strassen matrices for 2x2x2 (7 multiplications)
    # U matrix (2x2 per term): rows i=0,1; cols j=0,1
    # V matrix (2x2 per term): rows j=0,1; cols k=0,1  
    # W matrix (2x2 per term): rows i=0,1; cols k=0,1
    #
    # Standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    strassen_U = np.array([
        [1, 0, 1, 0],   # M1: A11 + A22
        [0, 0, 0, 1],   # M2: A21 + A22 (wait, let me reconsider)
        [1, 0, 0, 0],   # M3: A11
        [0, 0, 0, 1],   # M4: A22
        [1, 1, 0, 0],   # M5: A11 + A12
        [-1, 0, 1, 1],  # M6: A21 - A11 (hmm, this is getting complicated)
        [0, 1, -1, 0],  # M7: A12 - A22
    ])
    
    # Let me use the standard Strassen formulation more carefully.
    # For 2x2 matrices:
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    #
    # U factors (A side):
    # M1: (a11+a22) -> [1,0,1,0]
    # M2: (a21+a22) -> [0,0,1,1]  
    # M3: a11 -> [1,0,0,0]
    # M4: a22 -> [0,0,0,1]
    # M5: (a11+a12) -> [1,1,0,0]
    # M6: (a21-a11) -> [-1,0,1,0]
    # M7: (a12-a22) -> [0,1,-1,0]
    
    strassen_U = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0],
    ])
    
    # V factors (B side):
    # M1: (b11+b22) -> [1,0,0,1]
    # M2: b11 -> [1,0,0,0]
    # M3: (b21-b22) -> [0,1,-1,0]
    # M4: (b12-b22) -> [0,1,-1,0] wait no
    # M4: b12 - b22 -> [0,1,-1,0]
    # M5: b22 -> [0,0,0,1]
    # M6: (b11+b12) -> [1,1,0,0]
    # M7: (b21+b22) -> [0,1,0,1]
    
    strassen_V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, -1, 0],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ])
    
    # W factors (C side):
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each output position, which M's contribute:
    # C11: M1(+1), M4(+1), M5(-1), M7(+1)
    # C12: M3(+1), M5(+1)
    # C21: M2(+1), M4(+1)
    # C22: M1(+1), M2(-1), M3(+1), M6(+1)
    
    strassen_W = np.zeros((7, 4))
    # C11 is at index 0 (i=0, k=0)
    strassen_W[0, 0] = 1   # M1
    strassen_W[3, 0] = 1   # M4
    strassen_W[4, 0] = -1  # M5
    strassen_W[6, 0] = 1   # M7
    # C12 is at index 1 (i=0, k=1)
    strassen_W[2, 1] = 1   # M3
    strassen_W[4, 1] = 1   # M5
    # C21 is at index 2 (i=1, k=0)
    strassen_W[1, 2] = 1   # M2
    strassen_W[3, 2] = 1   # M4
    # C22 is at index 3 (i=1, k=1)
    strassen_W[0, 3] = 1   # M1
    strassen_W[1, 3] = -1  # M2
    strassen_W[2, 3] = 1   # M3
    strassen_W[5, 3] = 1   # M6

    for j_block in range(m // 2):
        for k_block in range(p // 2):
            for t in range(7):
                if time.time() > deadline:
                    return {"U": [list(r) for r in U_np], "V": [list(r) for r in V_np], "W": [list(r) for r in W_np]}

                # U: rows are i=1,2 (global i=1+li), cols are j in block
                for li in range(2):
                    for lj in range(2):
                        val = strassen_U[t, li * 2 + lj]
                        if val != 0:
                            gi = 1 + li
                            gj = j_block * 2 + lj
                            U_np[(gi * m + gj), col] += val

                # V: rows are j in block, cols are k in block
                for lj in range(2):
                    for lk in range(2):
                        val = strassen_V[t, lj * 2 + lk]
                        if val != 0:
                            gj = j_block * 2 + lj
                            gk = k_block * 2 + lk
                            V_np[(gj * p + gk), col] += val

                # W: rows are k in block, cols are i=1,2 (global i=1+li)
                for li in range(2):
                    for lk in range(2):
                        val = strassen_W[t, li * 2 + lk]
                        if val != 0:
                            gi = 1 + li
                            gk = k_block * 2 + lk
                            W_np[(gk * n + gi), col] += val

                col += 1

    U = [list(r) for r in U_np]
    V = [list(r) for r in V_np]
    W = [list(r) for r in W_np]

    return {"U": U, "V": V, "W": W}
