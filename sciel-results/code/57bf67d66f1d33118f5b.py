import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 54
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define Strassen blocks
    # j pairs: (0,1), (2,3)
    # k pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            # 7 Strassen terms
            
            # M1: (A0,j1 + A1,j2) * (Bj1,k1 + Bj2,k2)
            # U: (0,j1)=1, (1,j2)=1
            # V: (j1,k1)=1, (j2,k2)=1
            # W: (k1,0)=1, (k2,1)=1
            U[0*m + j1][col] = 1
            U[1*m + j2][col] = 1
            V[j1*p + k1][col] = 1
            V[j2*p + k2][col] = 1
            W[k1*n + 0][col] = 1
            W[k2*n + 1][col] = 1
            col += 1
            
            # M2: (A1,j1 + A1,j2) * Bj1,k1
            # U: (1,j1)=1, (1,j2)=1
            # V: (j1,k1)=1
            # W: (k1,1)=1
            U[1*m + j1][col] = 1
            U[1*m + j2][col] = 1
            V[j1*p + k1][col] = 1
            W[k1*n + 1][col] = 1
            col += 1
            
            # M3: A0,j1 * (Bj1,k2 - Bj2,k2)
            # U: (0,j1)=1
            # V: (j1,k2)=1, (j2,k2)=-1
            # W: (k2,0)=1
            U[0*m + j1][col] = 1
            V[j1*p + k2][col] = 1
            V[j2*p + k2][col] = -1
            W[k2*n + 0][col] = 1
            col += 1
            
            # M4: A1,j2 * (Bj2,k1 - Bj1,k1)
            # U: (1,j2)=1
            # V: (j2,k1)=1, (j1,k1)=-1
            # W: (k1,1)=1
            U[1*m + j2][col] = 1
            V[j2*p + k1][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + 1][col] = 1
            col += 1
            
            # M5: (A0,j1 + A0,j2) * Bj2,k2
            # U: (0,j1)=1, (0,j2)=1
            # V: (j2,k2)=1
            # W: (k2,0)=-1, (k2,1)=1  (from C00=-M5, C11=M5? No, C11 is C_{1, k2}? No, C_{1, k2} is M1-M2+M3+M6. C_{0, k2} is M3+M5. C_{0, k1} is M1+M4-M5+M7. C_{1, k1} is M2+M4.)
            # Let's re-verify W contributions for M5.
            # C_{0, k1} (C11): -1
            # C_{0, k2} (C12): +1
            # C_{1, k1} (C21): 0
            # C_{1, k2} (C22): 0
            W[k1*n + 0][col] = -1
            W[k2*n + 0][col] = 1
            col += 1
            
            # M6: (A1,j1 - A0,j1) * (Bj1,k1 + Bj1,k2)
            # U: (1,j1)=1, (0,j1)=-1
            # V: (j1,k1)=1, (j1,k2)=1
            # W: (k2,1)=1 (from C22=M6)
            # Check C22 formula: C_{1, k2} = M1 - M2 + M3 + M6.
            # So W_{k2, 1} = 1.
            U[1*m + j1][col] = 1
            U[0*m + j1][col] = -1
            V[j1*p + k1][col] = 1
            V[j1*p + k2][col] = 1
            W[k2*n + 1][col] = 1
            col += 1
            
            # M7: (A0,j2 - A1,j2) * (Bj2,k1 + Bj2,k2)
            # U: (0,j2)=1, (1,j2)=-1
            # V: (j2,k1)=1, (j2,k2)=1
            # W: (k1,0)=1 (from C11=M7)
            # Check C11 formula: C_{0, k1} = M1 + M4 - M5 + M7.
            # So W_{k1, 0} = 1.
            U[0*m + j2][col] = 1
            U[1*m + j2][col] = -1
            V[j2*p + k1][col] = 1
            V[j2*p + k2][col] = 1
            W[k1*n + 0][col] = 1
            col += 1
            
    # Remainder: j=4
    # i in 0,1; k in 0..5
    for i in range(n):
        for k in range(p):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
