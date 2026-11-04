import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # We will use 5 blocks of 2x3x2
    # Rank per block: 7 (Strassen) + 4 (linear) = 11
    # Total rank: 5 * 11 = 55
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B11
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B00
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Indices in A (2x3): 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2)
    # Indices in B (3x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(2,0), 5=(2,1)
    # Indices in C (2x2): 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # For each block t (0 to 4):
    # B cols: 2t, 2t+1
    # C cols: 2t, 2t+1
    
    for t in range(5):
        base_col = t * 11
        
        # Strassen part (uses A cols 0,1 and B rows 0,1)
        # Term 1: M1
        U[0][base_col + 0] += 1 # A00
        U[4][base_col + 0] += 1 # A11
        V[0][base_col + 0] += 1 # B00
        V[3][base_col + 0] += 1 # B11
        W[2*t][base_col + 0] += 1 # C00
        W[3+2*t][base_col + 0] += 1 # C11

        # Term 2: M2
        U[3][base_col + 1] += 1 # A10
        U[4][base_col + 1] += 1 # A11
        V[3][base_col + 1] += 1 # B11
        W[2+2*t][base_col + 1] += 1 # C10
        W[3+2*t][base_col + 1] += 1 # C11

        # Term 3: M3
        U[0][base_col + 2] += 1 # A00
        V[1][base_col + 2] += 1 # B01
        V[3][base_col + 2] -= 1 # B11
        W[1+2*t][base_col + 2] += 1 # C01

        # Term 4: M4
        U[4][base_col + 3] += 1 # A11
        V[2][base_col + 3] += 1 # B10
        V[0][base_col + 3] -= 1 # B00
        W[0+2*t][base_col + 3] -= 1 # C00
        W[2+2*t][base_col + 3] += 1 # C10

        # Term 5: M5
        U[0][base_col + 4] += 1 # A00
        U[1][base_col + 4] += 1 # A01
        V[0][base_col + 4] += 1 # B00
        W[0+2*t][base_col + 4] += 1 # C00
        W[1+2*t][base_col + 4] += 1 # C01

        # Term 6: M6
        U[3][base_col + 5] += 1 # A10
        U[0][base_col + 5] -= 1 # A00
        V[0][base_col + 5] += 1 # B00
        V[1][base_col + 5] += 1 # B01
        W[2+2*t][base_col + 5] += 1 # C10

        # Term 7: M7
        U[1][base_col + 6] += 1 # A01
        U[4][base_col + 6] -= 1 # A11
        V[2][base_col + 6] += 1 # B10
        V[3][base_col + 6] += 1 # B11
        W[0+2*t][base_col + 6] -= 1 # C00
        W[3+2*t][base_col + 6] += 1 # C11

        # Linear part (uses A col 2 and B row 2)
        # C00 += A02 B20
        U[2][base_col + 7] += 1 # A02
        V[4][base_col + 7] += 1 # B20
        W[0+2*t][base_col + 7] += 1 # C00

        # C01 += A02 B21
        U[2][base_col + 8] += 1 # A02
        V[5][base_col + 8] += 1 # B21
        W[1+2*t][base_col + 8] += 1 # C01

        # C10 += A12 B20
        U[5][base_col + 9] += 1 # A12
        V[4][base_col + 9] += 1 # B20
        W[2+2*t][base_col + 9] += 1 # C10

        # C11 += A12 B21
        U[5][base_col + 10] += 1 # A12
        V[5][base_col + 10] += 1 # B21
        W[3+2*t][base_col + 10] += 1 # C11

    return {"U": U, "V": V, "W": W}
