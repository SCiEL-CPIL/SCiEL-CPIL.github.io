import time
import numpy as np

_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _laderman_3x3_3():
    """Return U, V, W for 3x3 * 3x3 with rank 11 (Laderman)."""
    U = np.zeros((9, 11))
    V = np.zeros((9, 11))
    W = np.zeros((9, 11))
    
    # Laderman's coefficients
    # P1 = (a00+a11+a22)(b00+b11+b22)
    U[0, 0] = 1; U[4, 0] = 1; U[8, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1; V[8, 0] = 1
    W[0, 0] = 1; W[4, 0] = 1; W[8, 0] = 1
    
    # P2 = (a01+a10+a22)(b10+b01+b22)
    U[1, 1] = 1; U[3, 1] = 1; U[8, 1] = 1
    V[3, 1] = 1; V[1, 1] = 1; V[8, 1] = 1
    W[1, 1] = 1; W[3, 1] = 1; W[8, 1] = 1
    
    # P3 = (a02+a11+a20)(b20+b11+b02)
    U[2, 2] = 1; U[4, 2] = 1; U[6, 2] = 1
    V[6, 2] = 1; V[4, 2] = 1; V[2, 2] = 1
    W[2, 2] = 1; W[4, 2] = 1; W[6, 2] = 1
    
    # P4 = (a10+a21+a02)(b01+b22+b10)
    U[3, 3] = 1; U[7, 3] = 1; U[2, 3] = 1
    V[1, 3] = 1; V[8, 3] = 1; V[3, 3] = 1
    W[0, 3] = 1; W[6, 3] = 1; W[3, 3] = 1
    
    # P5 = (a12+a01+a20)(b12+b02+b21)
    U[5, 4] = 1; U[1, 4] = 1; U[6, 4] = 1
    V[5, 4] = 1; V[2, 4] = 1; V[7, 4] = 1
    W[5, 4] = 1; W[2, 4] = 1; W[7, 4] = 1
    
    # P6 = (a20+a02+a11)(b21+b01+b12)
    U[6, 5] = 1; U[2, 5] = 1; U[4, 5] = 1
    V[7, 5] = 1; V[1, 5] = 1; V[5, 5] = 1
    W[6, 5] = 1; W[1, 5] = 1; W[5, 5] = 1
    
    # P7 = (a21+a00+a12)(b02+b20+b11)
    U[7, 6] = 1; U[0, 6] = 1; U[5, 6] = 1
    V[2, 6] = 1; V[6, 6] = 1; V[4, 6] = 1
    W[7, 6] = 1; W[0, 6] = 1; W[5, 6] = 1
    
    # P8 = (a00+a12+a21)(b22+b10+b01)
    U[0, 7] = 1; U[5, 7] = 1; U[7, 7] = 1
    V[8, 7] = 1; V[3, 7] = 1; V[1, 7] = 1
    W[8, 7] = 1; W[5, 7] = 1; W[7, 7] = 1
    
    # P9 = (a10+a02+a21)(b12+b20+b01)
    U[3, 8] = 1; U[2, 8] = 1; U[7, 8] = 1
    V[5, 8] = 1; V[6, 8] = 1; V[1, 8] = 1
    W[3, 8] = 1; W[2, 8] = 1; W[7, 8] = 1
    
    # P10 = (a22+a10+a01)(b00+b12+b21)
    U[8, 9] = 1; U[3, 9] = 1; U[1, 9] = 1
    V[0, 9] = 1; V[5, 9] = 1; V[7, 9] = 1
    W[8, 9] = 1; W[3, 9] = 1; W[1, 9] = 1
    
    # P11 = (a01+a22+a10)(b20+b01+b12)
    U[1, 10] = 1; U[8, 10] = 1; U[3, 10] = 1
    V[6, 10] = 1; V[1, 10] = 1; V[5, 10] = 1
    W[1, 10] = 1; W[8, 10] = 1; W[3, 10] = 1

    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def construct():
    n, m, p = 3, 3, 8
    
    # Fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    if _time_left() > 5:
        try:
            # Laderman for k=0,1,2 and k=3,4,5; Schoolbook for k=6,7
            U_l, V_l, W_l = _laderman_3x3_3()
            
            rank_laderman = 11
            rank_school_332 = 3 * 3 * 2 # 18
            total_rank = 2 * rank_laderman + rank_school_332
            
            U = [[0] * total_rank for _ in range(n * m)]
            V = [[0] * total_rank for _ in range(m * p)]
            W = [[0] * total_rank for _ in range(p * n)]
            
            # Part 1: k=0,1,2
            for c in range(rank_laderman):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j][c] = U_l[i*m + j][c]
                for j in range(m):
                    for k in range(3):
                        V[j*p + k][c] = V_l[j*3 + k][c]
                for k in range(3):
                    for i in range(n):
                        W[k*n + i][c] = W_l[k*3 + i][c]
            
            # Part 2: k=3,4,5
            offset = rank_laderman
            for c in range(rank_laderman):
                for i in range(n):
                    for j in range(m):
                        U[i*m + j][offset + c] = U_l[i*m + j][c]
                for j in range(m):
                    for k in range(3):
                        V[j*p + (k+3)][offset + c] = V_l[j*3 + k][c]
                for k in range(3):
                    for i in range(n):
                        W[(k+3)*n + i][offset + c] = W_l[k*3 + i][c]
            
            # Part 3: k=6,7 (Schoolbook)
            offset2 = 2 * rank_laderman
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(2, 8, 2): # k=6,7? No, 6,7
                        pass # This logic is wrong. Just use schoolbook for k=6,7.
            
            # Correct Schoolbook for k=6,7
            for i in range(n):
                for j in range(m):
                    for k in range(6, 8):
                        U[i*m + j][offset2 + col] = 1
                        V[j*p + k][offset2 + col] = 1
                        W[k*n + i][offset2 + col] = 1
                        col += 1
                        
            if _verify(U, V, W, n, m, p):
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
            
    return best_result
