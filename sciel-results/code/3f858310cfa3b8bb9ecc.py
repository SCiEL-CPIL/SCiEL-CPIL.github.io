import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition coefficients
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k
    # W indices: (k, i) -> k*n + i
    
    # Strassen terms for A (2x2)
    # P1: (a+d) -> U[0,0]=1, U[1,1]=1
    # P2: (c+d) -> U[1,0]=1, U[1,1]=1
    # P3: a     -> U[0,0]=1
    # P4: d     -> U[1,1]=1
    # P5: (a+b) -> U[0,0]=1, U[0,1]=1
    # P6: (c-a) -> U[1,0]=1, U[0,0]=-1
    # P7: (b-d) -> U[0,1]=1, U[1,1]=-1
    
    strassen_U = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    
    # W coefficients for C (2x2)
    # C00: P1+1, P4+1, P5-1, P7+1
    # C01: P3+1, P5+1
    # C10: P2+1, P4+1
    # C11: P1+1, P2-1, P3+1, P6+1
    
    strassen_W = [
        [(0, 1), (3, 1)],   # P1
        [(1, 1), (3, -1)],  # P2
        [(2, 1), (3, 1)],   # P3
        [(0, 1), (1, 1)],   # P4
        [(0, -1), (2, 1)],  # P5
        [(3, 1)],           # P6
        [(0, 1)]            # P7
    ]
    
    def add_strassen_block(start_col, k_start):
        for r in range(7):
            col = start_col + r
            # Add U
            for idx, val in strassen_U[r]:
                U[idx][col] += val
            # Add W
            # k_start is 0, 2, 5
            # W index = k*n + i
            # k values are k_start and k_start+1
            # i values are 0 and 1
            # C indices in strassen_W are relative (0..3)
            # 0 -> C00 (i=0, k=k_start) -> idx k_start*n + 0
            # 1 -> C10 (i=1, k=k_start) -> idx k_start*n + 1
            # 2 -> C01 (i=0, k=k_start+1) -> idx (k_start+1)*n + 0
            # 3 -> C11 (i=1, k=k_start+1) -> idx (k_start+1)*n + 1
            
            for rel_idx, val in strassen_W[r]:
                i = rel_idx % 2
                k_rel = rel_idx // 2
                k_abs = k_start + k_rel
                w_idx = k_abs * n + i
                W[w_idx][col] += val
            
            # Add V
            # V index = j*p + k
            # P1: B00, B11 -> j=0,k=k_start; j=1,k=k_start+1
            # P2: B00      -> j=0,k=k_start
            # P3: B01, -B11-> j=0,k=k_start+1; j=1,k=k_start+1
            # P4: B10, -B00-> j=1,k=k_start; j=0,k=k_start
            # P5: B11      -> j=1,k=k_start+1
            # P6: B00, B01 -> j=0,k=k_start; j=0,k=k_start+1
            # P7: B10, B11 -> j=1,k=k_start; j=1,k_start+1
            
            if r == 0:
                V[0 * p + k_start][col] += 1
                V[1 * p + (k_start + 1)][col] += 1
            elif r == 1:
                V[0 * p + k_start][col] += 1
            elif r == 2:
                V[0 * p + (k_start + 1)][col] += 1
                V[1 * p + (k_start + 1)][col] += -1
            elif r == 3:
                V[1 * p + k_start][col] += 1
                V[0 * p + k_start][col] += -1
            elif r == 4:
                V[1 * p + (k_start + 1)][col] += 1
            elif r == 5:
                V[0 * p + k_start][col] += 1
                V[0 * p + (k_start + 1)][col] += 1
            elif r == 6:
                V[1 * p + k_start][col] += 1
                V[1 * p + (k_start + 1)][col] += 1

    # Block 1: k=0,1
    add_strassen_block(0, 0)
    # Block 2: k=2,3
    add_strassen_block(7, 2)
    
    # Block 3: k=4 (Schoolbook)
    # Cols 14-17
    # r=0: i=0, j=0
    U[0 * m + 0][14] = 1
    V[0 * p + 4][14] = 1
    W[4 * n + 0][14] = 1
    
    # r=1: i=0, j=1
    U[0 * m + 1][15] = 1
    V[1 * p + 4][15] = 1
    W[4 * n + 0][15] = 1
    
    # r=2: i=1, j=0
    U[1 * m + 0][16] = 1
    V[0 * p + 4][16] = 1
    W[4 * n + 1][16] = 1
    
    # r=3: i=1, j=1
    U[1 * m + 1][17] = 1
    V[1 * p + 4][17] = 1
    W[4 * n + 1][17] = 1
    
    # Block 4: k=5,6
    add_strassen_block(18, 5)

    return {"U": U, "V": V, "W": W}
