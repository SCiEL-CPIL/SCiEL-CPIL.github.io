import time
import numpy as np

DIMENSIONS = [2, 3, 6]
_START = time.time()
_DEADLINE = 120.0
_MARGIN = 5.0

def _remaining():
    return _DEADLINE - _MARGIN - (time.time() - _START)

def _build_rank_11_2x3x2():
    n, m, p = 2, 3, 2
    rank = 11
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen (r=0..6)
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[2][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[3][1] = 1
    V[0][1] = 1; V[2][1] = 1
    W[0][1] = -1; W[1][1] = 1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = -1
    # M4
    U[4][3] = 1
    V[1][3] = 1; V[3][3] = -1
    W[2][3] = 1; W[3][3] = -1
    # M5
    U[3][4] = 1; U[4][4] = 1
    V[2][4] = 1; V[3][4] = 1
    W[1][4] = 1; W[3][4] = -1
    # M6
    U[1][5] = 1; U[4][5] = -1
    V[2][5] = 1; V[3][5] = 1
    W[3][5] = 1
    # M7
    U[0][6] = 1; U[1][6] = -1
    V[0][6] = 1; V[1][6] = 1
    W[0][6] = 1
    
    # Extra terms for j=2
    # r=7: A02 B20 -> C00
    U[2][7] = 1
    V[4][7] = 1
    W[0][7] = 1
    # r=8: A02 B21 -> C01
    U[2][8] = 1
    V[5][8] = 1
    W[2][8] = 1
    # r=9: A12 B20 -> C10
    U[5][9] = 1
    V[4][9] = 1
    W[1][9] = 1
    # r=10: A12 B21 -> C11
    U[5][10] = 1
    V[5][10] = 1
    W[3][10] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Combine 3 blocks
    # Block 1: k=0,1
    # Block 2: k=2,3
    # Block 3: k=4,5
    
    U_blocks, V_blocks, W_blocks = [], [], []
    
    for b in range(3):
        Ub, Vb, Wb = _build_rank_11_2x3x2()
        # Shift V indices
        # V[j*p+k] -> V[j*6 + (k+2b)]
        Vb_shifted = [[0.0] * 11 for _ in range(18)]
        for j in range(3):
            for k in range(2):
                for r in range(11):
                    Vb_shifted[j*6 + (k + 2*b)][r] = Vb[j*2 + k][r]
        
        # Shift W indices
        # W[k*n+i] -> W[(k+2b)*2 + i]
        Wb_shifted = [[0.0] * 11 for _ in range(12)]
        for k in range(2):
            for i in range(2):
                for r in range(11):
                    Wb_shifted[(k + 2*b)*2 + i][r] = Wb[k*2 + i][r]
        
        U_blocks.append(Ub)
        V_blocks.append(Vb_shifted)
        W_blocks.append(Wb_shifted)
        
    # Concatenate
    U = []
    for i in range(6):
        U.append([0.0] * 33)
    for b in range(3):
        for i in range(6):
            for r in range(11):
                U[i][b*11 + r] = U_blocks[b][i][r]
                
    V = []
    for j in range(18):
        V.append([0.0] * 33)
    for b in range(3):
        for j in range(18):
            for r in range(11):
                V[j][b*11 + r] = V_blocks[b][j][r]
                
    W = []
    for k in range(12):
        W.append([0.0] * 33)
    for b in range(3):
        for k in range(12):
            for r in range(11):
                W[k][b*11 + r] = W_blocks[b][k][r]
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    print(construct())
