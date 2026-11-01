import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    n, m, p = 2, 2, 2
    rank = 7
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's 7 scalar products for 2x2 matrix multiplication
    # A = [[a00, a01], [a10, a11]] -> indices 0, 1, 2, 3
    # B = [[b00, b01], [b10, b11]] -> indices 0, 1, 2, 3
    # C = [[c00, c01], [c10, c11]]
    # W index convention: W[k*n + i]
    # c00 -> k=0, i=0 -> idx 0
    # c01 -> k=1, i=0 -> idx 2
    # c10 -> k=0, i=1 -> idx 1
    # c11 -> k=1, i=1 -> idx 3
    
    # M1: m0 = (a00 + a11) * (b00 + b11)
    # c00 += m0, c11 += m0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: m1 = (a10 + a11) * b00
    # c10 += m1, c11 -= m1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # M3: m2 = a00 * (b01 - b11)
    # c01 += m2, c11 += m2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # M4: m3 = a11 * (b10 - b00)
    # c00 += m3, c10 += m3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # M5: m4 = (a00 + a01) * b11
    # c00 -= m4, c01 += m4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: m5 = (a10 - a00) * (b00 + b01)
    # c11 += m5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7: m6 = (a01 - a11) * (b10 + b11)
    # c00 += m6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
