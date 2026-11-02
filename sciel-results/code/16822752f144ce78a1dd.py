import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    rank = 66
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    strassen_U = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,1,0],
        [0,1,0,1],
    ]
    strassen_W = [
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [1,1,0,0],
        [-1,0,1,0],
        [0,0,0,1],
        [1,0,0,1],
    ]
    
    col = 0
    i_blocks = [[0], [1, 2]]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                if len(ib) == 1:
                    i0 = ib[0]
                    for j0 in jb:
                        for k0 in kb:
                            U[i0 * m + j0][col] = 1
                            V[j0 * p + k0][col] = 1
                            W[k0 * n + i0][col] = 1
                            col += 1
                else:
                    i0, i1 = ib[0], ib[1]
                    j0, j1 = jb[0], jb[1]
                    k0, k1 = kb[0], kb[1]
                    for r in range(7):
                        for a in range(2):
                            for b in range(2):
                                val = strassen_U[r][a*2+b]
                                if val != 0:
                                    ii = i0 if a == 0 else i1
                                    jj = j0 if b == 0 else j1
                                    U[ii * m + jj][col] += val
                        for b in range(2):
                            for c in range(2):
                                val = strassen_V[r][b*2+c]
                                if val != 0:
                                    jj = j0 if b == 0 else j1
                                    kk = k0 if c == 0 else k1
                                    V[jj * p + kk][col] += val
                        for c in range(2):
                            for a in range(2):
                                val = strassen_W[r][c*2+a]
                                if val != 0:
                                    kk = k0 if c == 0 else k1
                                    ii = i0 if a == 0 else i1
                                    W[kk * n + ii][col] += val
                        col += 1
    
    return {"U": U, "V": V, "W": W}
