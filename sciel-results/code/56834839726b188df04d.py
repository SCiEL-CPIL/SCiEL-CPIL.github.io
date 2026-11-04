import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B10-B11)
    # M4: A11(B01-B00)
    # M5: (A00+A10)B11
    # M6: (A10-A00)(B00+B10)
    # M7: (A01-A11)(B11+B01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_U = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 0], [1, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    strassen_V = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 0], [1, -1]],# M3
        [[0, -1], [0, 1]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [0, 1]]  # M7
    ]
    strassen_W = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 1], [0, 0]], # M2 -> C10
        [[0, 0], [1, 0]], # M3 -> C01, C11
        [[1, 0], [0, 1]], # M4 -> C00, C10
        [[0, -1], [0, 1]],# M5 -> C01, C11
        [[0, 0], [0, 1]], # M6 -> C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    # Note: W indices are (k, i).
    # C00: i=0, k=0 -> W[0,0]
    # C01: i=0, k=1 -> W[1,0]
    # C10: i=1, k=0 -> W[0,1]
    # C11: i=1, k=1 -> W[1,1]
    
    # Let's re-verify W for Strassen
    # M1: C00(+1), C11(+1). W[0,0]=1, W[1,1]=1.
    # M2: C10(+1). W[0,1]=1.
    # M3: C01(+1), C11(+1). W[1,0]=1, W[1,1]=1.
    # M4: C00(+1), C10(+1). W[0,0]=1, W[0,1]=1.
    # M5: C01(-1), C11(-1). W[1,0]=-1, W[1,1]=-1.
    # M6: C11(+1). W[1,1]=1.
    # M7: C00(+1). W[0,0]=1.
    
    strassen_W_corrected = [
        [[1, 0], [0, 1]], # M1
        [[0, 1], [0, 0]], # M2
        [[0, 1], [1, 1]], # M3
        [[1, 1], [0, 0]], # M4
        [[0, -1], [0, -1]],# M5
        [[0, 0], [0, 1]], # M6
        [[1, 0], [0, 0]]  # M7
    ]
    
    # Helper to add a block
    def add_block(s, is_last=False):
        nonlocal col
        # s is 0, 1, 2 for 3x2 blocks
        # k_L = 2s, k_R = 2s+1
        k_L = 2 * s
        k_R = 2 * s + 1
        
        if not is_last:
            # 2x2x2 part (Strassen)
            for l in range(7):
                # U
                for i in range(2):
                    for j in range(2):
                        U[i * m + j][col] = strassen_U[l][i][j]
                # V
                for j in range(2):
                    for k in range(2):
                        V[j * p + (k_L + k)][col] = strassen_V[l][j][k]
                # W
                for k in range(2):
                    for i in range(2):
                        W[(k_L + k) * n + i][col] = strassen_W_corrected[l][k][i]
                col += 1
            
            # 2x1x2 part (A_R B_bot)
            # A_R is col 2 of A. B_bot is row 2 of B block.
            # Terms: A[i,2] * B[2, k_L], A[i,2] * B[2, k_R] for i=0,1
            for i in range(2):
                for k in [k_L, k_R]:
                    U[i * m + 2][col] = 1
                    V[2 * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        else:
            # 3x1 block (B_4)
            # k = 6
            k = 6
            # 2x3x1 part
            # Terms: A[i,j] * B[j, k] for i=0,1, j=0,1,2
            for i in range(2):
                for j in range(3):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    # Add 3 blocks of 3x2
    for s in range(3):
        add_block(s)
    
    # Add 1 block of 3x1
    add_block(3, is_last=True)
    
    return {"U": U, "V": V, "W": W}
