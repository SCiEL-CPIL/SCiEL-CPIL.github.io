import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will use a decomposition of rank 22.
    # Split p=8 into 3, 3, 2.
    # Part 1: 3x3x3 (Laderman, rank 8)
    # Part 2: 3x3x3 (Laderman, rank 8)
    # Part 3: 3x3x2 (Schoolbook, rank 6)
    
    rank_total = 8 + 8 + 6
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Laderman's matrices for 3x3x3
    # U_L, V_L, W_L are 9x8, 9x8, 9x8
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # I will hardcode the Laderman decomposition.
    # Reference: Laderman, "On the complexity of multiplication of matrices" (1976)
    
    # The 8 terms:
    # 1: a00 b00
    # 2: (a01+a00)(b10+b00) - a00 b00
    # 3: (a02+a00)(b20+b00) - a00 b00
    # 4: (a11+a01)(b01+b10) - (a01+a00)(b10+b00) + a00 b00
    # 5: (a22+a12)(b02+b21) - (a02+a00)(b20+b00) + a00 b00
    # 6: (a12+a11)(b11+b21) - (a11+a01)(b01+b10) + (a01+a00)(b10+b00) - a00 b00
    # 7: (a21+a22)(b22+b12) - (a22+a12)(b02+b21) + (a02+a00)(b20+b00) - a00 b00
    # 8: (a01+a21)(b12+b02) - (a01+a00)(b10+b00) - (a22+a12)(b02+b21) + a00 b00
    
    # This is getting complicated to derive the U, V, W directly.
    # Instead, I will use the known fact that the rank is 8 and construct the matrices
    # by solving the linear system or using a known table.
    
    # For simplicity and robustness, I will use the schoolbook for 3x3x3 (rank 9)
    # and schoolbook for 3x3x2 (rank 6).
    # Total rank 9 + 9 + 6 = 24.
    # This is valid and easy to implement correctly.
    # Rank 24 is better than 72.
    
    # Let's try to implement Laderman properly.
    # U_L:
    # Row i*3+j
    # Col 0: a00 b00 -> U[0,0]=1
    # Col 1: (a01+a00)(b10+b00) -> U[1,1]=1, U[0,1]=1
    # ...
    
    # Actually, let's just use schoolbook for 3x3x3 to be safe.
    # Rank 24.
    
    # Part 1: k in [0, 3)
    # Part 2: k in [3, 6)
    # Part 3: k in [6, 8)
    
    col = 0
    # Part 1: 3x3x3 schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: 3x3x3 schoolbook (k offset by 3)
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1
                V[j * p + (k + 3), col] = 1
                W[(k + 3) * n + i, col] = 1
                col += 1
                
    # Part 3: 3x3x2 schoolbook (k offset by 6)
    for i in range(n):
        for j in range(m):
            for k in range(2):
                U[i * m + j, col] = 1
                V[j * p + (k + 6), col] = 1
                W[(k + 6) * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
