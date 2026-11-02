import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # rank = 7 * 7 + 4 = 53
    rank = 53
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + j -> 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V indices: j*p + k. For a block of 2, k is local 0,1.
    # W indices: k*n + i. For a block of 2, k is local 0,1.
    
    # Strassen U vectors (for A)
    # U_c[i,j]
    strassen_U = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [0, 0, 1, -1],# M6: A10 - A00
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # Strassen V vectors (for B)
    # V_c[j,k] where k is local 0,1.
    # j=0,1; k=0,1.
    # Index in V for block b, local k: (j * 15) + (2*b + k)
    strassen_V = [
        [1, 1, 1, 1], # M1: B00 + B11 (j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1) -> B00+B01+B10+B11? No.
                      # M1 = (A00+A11)(B00+B11). B part is B00+B11.
                      # B00 is j=0,k=0. B11 is j=1,k=1.
                      # So V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=1.
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    # Let's re-verify M1 V vector.
    # M1 = (A00+A11)(B00+B11).
    # B00 is at j=0, k=0. B11 is at j=1, k=1.
    # So V[0,0]=1, V[0,1]=0, V[1,0]=0, V[1,1]=1.
    # My previous list had [1,1,1,1] for M1 which is wrong.
    
    strassen_V = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 1, -1],# M4: B10 - B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00 + B01
        [0, 0, 1, 1]  # M7: B10 + B11
    ]
    
    # Strassen W vectors (for C)
    # W_c[k,i] where k is local 0,1.
    # Index in W for block b, local k: (2*b + k) * 2 + i
    # C00 (i=0,k=0), C01 (i=0,k=1), C10 (i=1,k=0), C11 (i=1,k=1)
    # M1: C00 + C11. W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=1.
    # M2: C10. W[0,0]=0, W[0,1]=0, W[1,0]=1, W[1,1]=0.
    # M3: C01. W[0,0]=0, W[0,1]=1, W[1,0]=0, W[1,1]=0.
    # M4: C00 + C10. W[0,0]=1, W[0,1]=0, W[1,0]=1, W[1,1]=0.
    # M5: C00 - C11. W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=-1.
    # M6: C11. W[0,0]=0, W[0,1]=0, W[1,0]=0, W[1,1]=1.
    # M7: C00. W[0,0]=1, W[0,1]=0, W[1,0]=0, W[1,1]=0.
    
    strassen_W = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 0], # M2
        [0, 1, 0, 0], # M3
        [1, 0, 1, 0], # M4
        [1, 0, 0, -1],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Fill U, V, W for 7 Strassen blocks
    for b in range(7):
        for s in range(7):
            c = b * 7 + s
            # U
            for i in range(2):
                for j in range(2):
                    U[i*2+j][c] = strassen_U[s][i*2+j]
            # V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2*b + k_local
                    V[j*15+k_global][c] = strassen_V[s][j*2+k_local]
            # W
            for k_local in range(2):
                for i in range(2):
                    k_global = 2*b + k_local
                    W[k_global*2+i][c] = strassen_W[s][k_local*2+i]
                    
    # Fill for the last block (2x2x1)
    # Rank 4
    # K1 = A00 B00
    # K2 = A01 B10
    # K3 = A10 B00
    # K4 = A11 B10
    # C00 = K1 + K2
    # C10 = K3 + K4
    
    # U vectors
    last_U = [
        [1, 0, 0, 0], # K1: A00
        [0, 1, 0, 0], # K2: A01
        [0, 0, 1, 0], # K3: A10
        [0, 0, 0, 1]  # K4: A11
    ]
    
    # V vectors
    # B is 2x1. j=0,1; k=0.
    # B00 is j=0, k=0. B10 is j=1, k=0.
    # k_global = 14.
    last_V = [
        [1, 0], # K1: B00
        [0, 1], # K2: B10
        [1, 0], # K3: B00
        [0, 1]  # K4: B10
    ]
    
    # W vectors
    # C is 2x1. k=0; i=0,1.
    # C00 is k=0, i=0. C10 is k=0, i=1.
    last_W = [
        [1, 0], # K1: C00
        [1, 0], # K2: C00
        [0, 1], # K3: C10
        [0, 1]  # K4: C10
    ]
    
    for s in range(4):
        c = 49 + s
        for i in range(2):
            for j in range(2):
                U[i*2+j][c] = last_U[s][i*2+j]
        for j in range(2):
            k_global = 14
            V[j*15+k_global][c] = last_V[s][j]
        for i in range(2):
            k_global = 14
            W[k_global*2+i][c] = last_W[s][i]
            
    return {"U": U, "V": V, "W": W}
