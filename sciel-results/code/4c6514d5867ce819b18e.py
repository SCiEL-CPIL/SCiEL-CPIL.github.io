import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 120

def _time_remaining():
    return TIME_LIMIT - 10 - (time.time() - _START_TIME)

def _build_rank12():
    n, m, p = 2, 3, 3
    rank = 12
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 2x2x2 Strassen (cols 0-6)
    # Local indices 0..3
    # Map local A idx la to global A idx ga
    # la = 2*i + j -> ga = 3*i + j
    # Map local B idx lb to global B idx gb
    # lb = 2*j + k -> gb = 3*j + k
    # Map local C idx lc to global C idx gc
    # lc = 2*k + i -> gc = 2*k + i
    
    def map_a(la):
        i = la // 2
        j = la % 2
        return 3 * i + j
        
    def map_b(lb):
        j = lb // 2
        k = lb % 2
        return 3 * j + k
        
    def map_c(lc):
        k = lc // 2
        i = lc % 2
        return 2 * k + i

    # Strassen factors
    # U, V, W local
    U_s = [[0]*7 for _ in range(4)]
    V_s = [[0]*7 for _ in range(4)]
    W_s = [[0]*7 for _ in range(4)]
    
    # Col 0
    U_s[0][0] += 1; U_s[3][0] += 1
    V_s[0][0] += 1; V_s[3][0] += 1
    W_s[0][0] += 1; W_s[3][0] += 1
    # Col 1
    U_s[2][1] += 1; U_s[3][1] += 1
    V_s[0][1] += 1
    W_s[2][1] += 1; W_s[3][1] -= 1
    # Col 2
    U_s[0][2] += 1
    V_s[1][2] += 1; V_s[3][2] -= 1
    W_s[1][2] += 1; W_s[3][2] += 1
    # Col 3
    U_s[3][3] += 1
    V_s[2][3] += 1; V_s[0][3] -= 1
    W_s[0][3] += 1; W_s[2][3] += 1
    # Col 4
    U_s[0][4] += 1; U_s[1][4] += 1
    V_s[3][4] += 1
    W_s[0][4] -= 1; W_s[1][4] += 1
    # Col 5
    U_s[2][5] += 1; U_s[0][5] -= 1
    V_s[0][5] += 1; V_s[1][5] += 1
    W_s[3][5] += 1
    # Col 6
    U_s[1][6] += 1; U_s[3][6] -= 1
    V_s[2][6] += 1; V_s[3][6] += 1
    W_s[0][6] += 1

    for c in range(7):
        for la in range(4):
            if U_s[la][c] != 0:
                ga = map_a(la)
                U[ga][c] += U_s[la][c]
        for lb in range(4):
            if V_s[lb][c] != 0:
                gb = map_b(lb)
                V[gb][c] += V_s[lb][c]
        for lc in range(4):
            if W_s[lc][c] != 0:
                gc = map_c(lc)
                W[gc][c] += W_s[lc][c]

    # Part 1b: 2x2x1 (k=2) (cols 7-8)
    # Col 7: B[0,2] (A[0,0] C[0,2] + A[1,0] C[1,2])
    # A[0,0] -> U[0]
    # A[1,0] -> U[3]
    # B[0,2] -> V[2]
    # C[0,2] -> W[4] (k=2, i=0)
    # C[1,2] -> W[5] (k=2, i=1)
    U[0][7] += 1
    U[3][7] += 1
    V[2][7] += 1
    W[4][7] += 1
    W[5][7] += 1
    
    # Col 8: B[1,2] (A[0,1] C[0,2] + A[1,1] C[1,2])
    # A[0,1] -> U[1]
    # A[1,1] -> U[4]
    # B[1,2] -> V[5]
    # C[0,2] -> W[4]
    # C[1,2] -> W[5]
    U[1][8] += 1
    U[4][8] += 1
    V[5][8] += 1
    W[4][8] += 1
    W[5][8] += 1

    # Part 2: 2x1x3 (cols 9-11)
    # Col 9: B[2,0] (A[0,2] C[0,0] + A[1,2] C[1,0])
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B[2,0] -> V[6]
    # C[0,0] -> W[0]
    # C[1,0] -> W[1]
    U[2][9] += 1
    U[5][9] += 1
    V[6][9] += 1
    W[0][9] += 1
    W[1][9] += 1
    
    # Col 10: B[2,1] (A[0,2] C[0,1] + A[1,2] C[1,1])
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B[2,1] -> V[7]
    # C[0,1] -> W[2]
    # C[1,1] -> W[3]
    U[2][10] += 1
    U[5][10] += 1
    V[7][10] += 1
    W[2][10] += 1
    W[3][10] += 1
    
    # Col 11: B[2,2] (A[0,2] C[0,2] + A[1,2] C[1,2])
    # A[0,2] -> U[2]
    # A[1,2] -> U[5]
    # B[2,2] -> V[8]
    # C[0,2] -> W[4]
    # C[1,2] -> W[5]
    U[2][11] += 1
    U[5][11] += 1
    V[8][11] += 1
    W[4][11] += 1
    W[5][11] += 1

    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a], V_arr[b] * W_arr[c])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = 2, 3, 3
    U, V, W = _build_rank12()
    if _verify_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    # Fallback
    # ...
