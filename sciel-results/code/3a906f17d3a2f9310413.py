import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_28():
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def strassen_2x2(U_rows, V_rows, W_rows, offset):
        U_e = []
        V_e = []
        W_e = []
        col = offset
        
        # M1 = (A00+A11)*(B00+B11)
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], -1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M2 = (A10+A11)*B00
        U_e.append((U_rows[2], 1, col))
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M3 = A00*(B10-B11)
        U_e.append((U_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        V_e.append((V_rows[3], -1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[2], 1, col))
        col += 1
        
        # M4 = A11*(B01-B10)
        U_e.append((U_rows[3], 1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[1], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[3], 1, col))
        col += 1
        
        # M5 = (A00+A10)*B11
        U_e.append((U_rows[0], 1, col))
        U_e.append((U_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[2], -1, col))
        V_e.append((V_rows[0], 1, col))
        V_e.append((V_rows[1], 1, col))
        W_e.append((W_rows[0], -1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], 1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U_e.append((U_rows[1], 1, col))
        U_e.append((U_rows[3], -1, col))
        V_e.append((V_rows[2], 1, col))
        V_e.append((V_rows[3], 1, col))
        W_e.append((W_rows[0], 1, col))
        W_e.append((W_rows[1], 1, col))
        W_e.append((W_rows[2], -1, col))
        W_e.append((W_rows[3], -1, col))
        col += 1
        
        return U_e, V_e, W_e, col

    U_e1, V_e1, W_e1, _ = strassen_2x2([0,1,2,3], [0,1,2,3], [0,1,2,3], 0)
    U_e2, V_e2, W_e2, _ = strassen_2x2([4,5,6,7], [8,9,10,11], [0,1,2,3], 7)
    U_e3, V_e3, W_e3, _ = strassen_2x2([0,1,2,3], [12,13,14,15], [4,5,6,7], 14)
    U_e4, V_e4, W_e4, _ = strassen_2x2([4,5,6,7], [12,13,14,15], [4,5,6,7], 21)
    
    for row, val, col in U_e1 + U_e2 + U_e3 + U_e4:
        U[row][col] += val
    for row, val, col in V_e1 + V_e2 + V_e3 + V_e4:
        V[row][col] += val
    for row, val, col in W_e1 + W_e2 + W_e3 + W_e4:
        W[row][col] += val
        
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
                    max_err = max(max_err, abs(val - 1.0))
    return max_err

def construct():
    start_time = time.time()
    U, V, W = _build_strassen_28()
    err = _verify_decomposition(U, V, W)
    
    if err < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook rank 32
    n, m, p = 2, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
