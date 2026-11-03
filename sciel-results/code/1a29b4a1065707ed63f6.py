import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j1 in range(2):
        for i1 in range(2):
            for k1 in range(3):
                # Local Strassen decomposition for 2x2 * 2x2
                # i2, j2, k2 in {0, 1}
                # M1: A00 * (B10 + B11)
                # M2: (A00 + A01) * B10
                # M3: A11 * (B00 + B01)
                # M4: (A10 - A00) * B01
                # M5: (A01 + A11) * (B00 - B10)
                # M6: (A00 + A10) * (B11 - B01)
                # M7: (A10 + A11) * B00
                
                # Coefficients for C[i2, k2]
                # C00 = M1 - M4 + M5 + M7
                # C01 = M2 + M4
                # C10 = M3 + M6
                # C11 = M1 - M3 + M2 + M6
                
                coeffs = [
                    [1, 0, 0, -1, 1, 0, 1], # i2=0, k2=0
                    [0, 1, 0, 1, 0, 0, 0], # i2=0, k2=1
                    [0, 0, 1, 0, 0, 1, 0], # i2=1, k2=0
                    [1, 0, -1, 0, 0, 1, 0]  # i2=1, k2=1
                ]
                
                # U and V coefficients for each M_s
                # M1: A00 * (B10 + B11)
                # U: A00 -> (0,0)=1
                # V: B10+B11 -> (1,0)=1, (1,1)=1
                # M2: (A00+A01) * B10
                # U: A00+A01 -> (0,0)=1, (0,1)=1
                # V: B10 -> (1,0)=1
                # M3: A11 * (B00+B01)
                # U: A11 -> (1,1)=1
                # V: B00+B01 -> (0,0)=1, (0,1)=1
                # M4: (A10-A00) * B01
                # U: A10-A00 -> (1,0)=1, (0,0)=-1
                # V: B01 -> (0,1)=1
                # M5: (A01+A11) * (B00-B10)
                # U: A01+A11 -> (0,1)=1, (1,1)=1
                # V: B00-B10 -> (0,0)=1, (1,0)=-1
                # M6: (A00+A10) * (B11-B01)
                # U: A00+A10 -> (0,0)=1, (1,0)=1
                # V: B11-B01 -> (1,1)=1, (0,1)=-1
                # M7: (A10+A11) * B00
                # U: A10+A11 -> (1,0)=1, (1,1)=1
                # V: B00 -> (0,0)=1
                
                u_coeffs = [
                    {(0,0): 1},
                    {(0,0): 1, (0,1): 1},
                    {(1,1): 1},
                    {(1,0): 1, (0,0): -1},
                    {(0,1): 1, (1,1): 1},
                    {(0,0): 1, (1,0): 1},
                    {(1,0): 1, (1,1): 1}
                ]
                
                v_coeffs = [
                    {(1,0): 1, (1,1): 1},
                    {(1,0): 1},
                    {(0,0): 1, (0,1): 1},
                    {(0,1): 1},
                    {(0,0): 1, (1,0): -1},
                    {(1,1): 1, (0,1): -1},
                    {(0,0): 1}
                ]
                
                for s in range(7):
                    # Fill U
                    for (i2, j2), val in u_coeffs[s].items():
                        i = 2 * i1 + i2
                        j = 2 * j1 + j2
                        U[i * m + j][col] += val
                        
                    # Fill V
                    for (j2, k2), val in v_coeffs[s].items():
                        j = 2 * j1 + j2
                        k = 2 * k1 + k2
                        V[j * p + k][col] += val
                        
                    # Fill W
                    for i2 in range(2):
                        for k2 in range(2):
                            val = coeffs[i2][k2][s]
                            if val != 0:
                                i = 2 * i1 + i2
                                k = 2 * k1 + k2
                                W[k * n + i][col] += val
                                
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
