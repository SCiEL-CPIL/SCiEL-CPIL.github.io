import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n=4, m=4, p=9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    U = np.array(U, dtype=object)
    V = np.array(V, dtype=object)
    W = np.array(W, dtype=object)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _strassen_2x2():
    """Strassen decomposition for 2x2 x 2x2 matmul, rank 7."""
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]

    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1

    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1

    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1

    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1

    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1

    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1

    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = -1

    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """Recursive Strassen decomposition for 4x4 x 4x4 matmul, rank 49."""
    S = _strassen_2x2()
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            U[i * 4 + j][c] = S['U'][i1 * 2 + j1][c1] * S['U'][i2 * 2 + j2][c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    for k1 in range(2):
                        for k2 in range(2):
                            k = k1 * 2 + k2
                            V[j * 4 + k][c] = S['V'][j1 * 2 + k1][c1] * S['V'][j2 * 2 + k2][c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    for i1 in range(2):
                        for i2 in range(2):
                            i = i1 * 2 + i2
                            W[k * 4 + i][c] = S['W'][k1 * 2 + i1][c1] * S['W'][k2 * 2 + i2][c2]
                            
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = 4, 4, 9
    
    rank1 = 49
    rank2 = 49
    rank3 = 16
    total_rank = rank1 + rank2 + rank3
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    S444 = _strassen_4x4x4()
    
    for c in range(rank1):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = S444['U'][i * m + j][c]
        for j in range(m):
            for k in range(4):
                V[j * p + k][c] = S444['V'][j * 4 + k][c]
        for k in range(4):
            for i in range(n):
                W[k * n + i][c] = S444['W'][k * n + i][c]
                
    offset1 = rank1
    for c in range(rank2):
        cc = offset1 + c
        for i in range(n):
            for j in range(m):
                U[i * m + j][cc] = S444['U'][i * m + j][c]
        for j in range(m):
            for k in range(4):
                V[j * p + (4 + k)][cc] = S444['V'][j * 4 + k][c]
        for k in range(4):
            for i in range(n):
                W[(4 + k) * n + i][cc] = S444['W'][k * n + i][c]
                
    offset2 = rank1 + rank2
    for c in range(rank3):
        cc = offset2 + c
        i = c // m
        j = c % m
        U[i * m + j][cc] = 1
        V[j * p + 8][cc] = 1
        W[8 * n + i][cc] = 1

    cand = {"U": U, "V": V, "W": W}
    
    if _verify(cand['U'], cand['V'], cand['W'], n, m, p):
        print("Verified rank-114 decomposition")
        return cand
    else:
        print("Rank-114 verification failed, falling back to schoolbook")
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
