import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(ja, jb, ka, kb):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U[0*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[1*m+ja][col] = 1
        U[1*m+jb][col] = 1
        V[ja*p+ka][col] = 1
        W[ka*n+1][col] = 1
        W[kb*n+1][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[0*m+ja][col] = 1
        V[ja*p+kb][col] = 1
        V[jb*p+kb][col] = -1
        W[kb*n+0][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[1*m+jb][col] = 1
        V[jb*p+ka][col] = 1
        V[ja*p+ka][col] = -1
        W[ka*n+0][col] = 1
        W[ka*n+1][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[0*m+ja][col] = 1
        U[0*m+jb][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = -1
        W[kb*n+0][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[1*m+ja][col] = 1
        U[0*m+ja][col] = -1
        V[ja*p+ka][col] = 1
        V[ja*p+kb][col] = 1
        W[kb*n+1][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[0*m+jb][col] = 1
        U[1*m+jb][col] = -1
        V[jb*p+ka][col] = 1
        V[jb*p+kb][col] = 1
        W[ka*n+0][col] = 1
        col += 1

    # Apply Strassen to 4 blocks
    add_strassen_block(0, 1, 0, 1)
    add_strassen_block(2, 3, 0, 1)
    add_strassen_block(0, 1, 2, 3)
    add_strassen_block(2, 3, 2, 3)
    
    # Schoolbook for index 4
    for i in [0, 1]:
        for k in [0, 1, 2, 3]:
            U[i*m+4][col] = 1
            V[4*p+k][col] = 1
            W[k*n+i][col] = 1
            col += 1
            
    for i in [0, 1]:
        for j in [0, 1, 2, 3]:
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
            
    for i in [0, 1]:
        U[i*m+4][col] = 1
        V[4*p+4][col] = 1
        W[4*n+i][col] = 1
        col += 1
        
    if col != 46:
        # Fallback
        rank_school = n * m * p
        U_school = np.zeros((n * m, rank_school), dtype=float)
        V_school = np.zeros((m * p, rank_school), dtype=float)
        W_school = np.zeros((p * n, rank_school), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][c] = 1
                    V_school[j*p+k][c] = 1
                    W_school[k*n+i][c] = 1
                    c += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
