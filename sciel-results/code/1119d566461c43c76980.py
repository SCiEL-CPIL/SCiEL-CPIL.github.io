import time
import sys

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2 blocks
    # 8 blocks of 2 columns each
    for l in range(p // 2):
        for q in range(7):
            r = l * 7 + q
            
            # M1 = (A00 + A11)(B0k + B1k+1)
            if q == 0:
                U[0][r] = 1; U[3][r] = 1
                V[2*l][r] = 1; V[16+2*l+1][r] = 1
                W[4*l][r] = 1; W[4*l+3][r] = 1
            # M2 = (A10 + A11)B0k
            elif q == 1:
                U[2][r] = 1; U[3][r] = 1
                V[2*l][r] = 1
                W[4*l+1][r] = 1
            # M3 = A00(B0k+1 - B1k+1)
            elif q == 2:
                U[0][r] = 1
                V[2*l+1][r] = 1; V[16+2*l+1][r] = -1
                W[4*l+2][r] = 1; W[4*l+3][r] = 1
            # M4 = A11(B1k - B0k)
            elif q == 3:
                U[3][r] = 1
                V[16+2*l][r] = 1; V[2*l][r] = -1
                W[4*l][r] = 1; W[4*l+1][r] = 1
            # M5 = (A00 + A01)B1k+1
            elif q == 4:
                U[0][r] = 1; U[1][r] = 1
                V[16+2*l+1][r] = 1
                W[4*l][r] = -1; W[4*l+2][r] = 1
            # M6 = (A10 - A00)(B0k + B0k+1)
            elif q == 5:
                U[2][r] = 1; U[0][r] = -1
                V[2*l][r] = 1; V[2*l+1][r] = 1
                W[4*l+3][r] = 1
            # M7 = (A01 - A11)(B1k + B1k+1)
            elif q == 6:
                U[1][r] = 1; U[3][r] = -1
                V[16+2*l][r] = 1; V[16+2*l+1][r] = 1
                W[4*l][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
