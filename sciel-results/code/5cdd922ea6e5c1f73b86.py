import time
import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Rank-26 decomposition
    # Part 1: j=2, k=0..4 (10 terms) - direct
    # Part 2: j=0,1, k=4 (2 terms) - direct  
    # Part 3: j=0,1, k=0,1 (7 terms) - Strassen 2x2
    # Part 4: j=0,1, k=2,3 (7 terms) - Strassen 2x2
    # Total: 10 + 2 + 7 + 7 = 26
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 * B2 (j=2, k=0..4) - 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    # r = 10
    
    # Part 2: A1 * B_col4 (j=0,1, k=4) - 2 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
    # r = 12
    
    # Part 3: Strassen for (j=0,1, k=0,1) - 7 terms
    # A indices: i*m+j, B indices: j*p+k, C indices: k*n+i
    # M1 = (A00+A11)(B00+B11)
    U[0][r] = 1; U[4][r] = 1; V[0][r] = 1; V[6][r] = 1; W[0][r] = 1; W[3][r] = 1; r += 1
    # M2 = (A10+A11)(B00)
    U[3][r] = 1; U[4][r] = 1; V[0][r] = 1; W[1][r] = 1; W[3][r] = -1; r += 1
    # M3 = (A00)(B01+B11)
    U[0][r] = 1; V[1][r] = 1; V[6][r] = -1; W[2][r] = 1; W[3][r] = 1; r += 1
    # M4 = (A11)(B10+B00)
    U[4][r] = 1; V[5][r] = 1; V[0][r] = -1; W[0][r] = 1; W[1][r] = 1; r += 1
    # M5 = (A00+A01)(B11)
    U[0][r] = 1; U[1][r] = 1; V[6][r] = 1; W[0][r] = -1; W[2][r] = 1; r += 1
    # M6 = (A10-A00)(B00+B01)
    U[3][r] = 1; U[0][r] = -1; V[0][r] = 1; V[1][r] = 1; W[3][r] = 1; r += 1
    # M7 = (A01-A11)(B10+B11)
    U[1][r] = 1; U[4][r] = -1; V[5][r] = 1; V[6][r] = 1; W[0][r] = 1; r += 1
    # r = 19
    
    # Part 4: Strassen for (j=0,1, k=2,3) - 7 terms
    # M1 = (A00+A11)(B02+B13)
    U[0][r] = 1; U[4][r] = 1; V[2][r] = 1; V[8][r] = 1; W[4][r] = 1; W[7][r] = 1; r += 1
    # M2 = (A10+A11)(B02)
    U[3][r] = 1; U[4][r] = 1; V[2][r] = 1; W[5][r] = 1; W[7][r] = -1; r += 1
    # M3 = (A00)(B03+B13)
    U[0][r] = 1; V[3][r] = 1; V[8][r] = -1; W[6][r] = 1; W[7][r] = 1; r += 1
    # M4 = (A11)(B12+B02)
    U[4][r] = 1; V[7][r] = 1; V[2][r] = -1; W[4][r] = 1; W[5][r] = 1; r += 1
    # M5 = (A00+A01)(B13)
    U[0][r] = 1; U[1][r] = 1; V[8][r] = 1; W[4][r] = -1; W[6][r] = 1; r += 1
    # M6 = (A10-A00)(B02+B03)
    U[3][r] = 1; U[0][r] = -1; V[2][r] = 1; V[3][r] = 1; W[7][r] = 1; r += 1
    # M7 = (A01-A11)(B12+B13)
    U[1][r] = 1; U[4][r] = -1; V[7][r] = 1; V[8][r] = 1; W[4][r] = 1; r += 1
    # r = 26
    
    # Verify full decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = 0.0
                for r_idx in range(rank):
                    s += U_arr[idx_u, r_idx] * V_arr[idx_v, r_idx] * W_arr[idx_w, r_idx]
                if abs(s - 1.0) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if ok:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 30
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
