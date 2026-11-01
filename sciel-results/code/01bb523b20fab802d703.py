import numpy as np
import time

def construct():
    n, m, p = 2, 2, 12
    
    # Rank 42 solution: 6 blocks of 2x2 Strassen (rank 7 each)
    # Partition p=12 into 6 pairs.
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        base = t * 7
        for s in range(7):
            c = base + s
            # V indices: row in A (0..1) * p + col in B (0..11)
            # Block t covers cols 2*t and 2*t+1
            v_b00 = 0 * p + 2 * t
            v_b01 = 0 * p + 2 * t + 1
            v_b10 = 1 * p + 2 * t
            v_b11 = 1 * p + 2 * t + 1
            
            # W indices: row in C (0..11) * n + col in A (0..1)
            # Block t covers rows 2*t and 2*t+1 of C
            w_c00 = (2 * t) * n + 0
            w_c01 = (2 * t + 1) * n + 0
            w_c10 = (2 * t) * n + 1
            w_c11 = (2 * t + 1) * n + 1
            
            # Strassen decomposition for 2x2
            # A = [[a00, a01], [a10, a11]]
            # B = [[b00, b01], [b10, b11]]
            # C = A*B
            
            if s == 0:
                # M1 = (a00 + a10) * (b00 + b10)
                U[0][c] = 1; U[2][c] = 1
                V[v_b00][c] = 1; V[v_b10][c] = 1
                W[w_c00][c] = 1; W[w_c10][c] = 1
            elif s == 1:
                # M2 = a10 * (b01 - b00)
                U[2][c] = 1
                V[v_b01][c] = 1; V[v_b00][c] = -1
                W[w_c01][c] = 1; W[w_c11][c] = 1
            elif s == 2:
                # M3 = a00 * (b11 - b10)
                U[0][c] = 1
                V[v_b11][c] = 1; V[v_b10][c] = -1
                W[w_c00][c] = 1; W[w_c10][c] = 1
            elif s == 3:
                # M4 = (a01 + a11) * (b00 + b10)
                U[1][c] = 1; U[3][c] = 1
                V[v_b00][c] = 1; V[v_b10][c] = 1
                W[w_c01][c] = 1; W[w_c11][c] = 1
            elif s == 4:
                # M5 = (a00 - a10) * (b10 + b11)
                U[0][c] = 1; U[2][c] = -1
                V[v_b10][c] = 1; V[v_b11][c] = 1
                W[w_c00][c] = 1; W[w_c10][c] = -1
            elif s == 5:
                # M6 = (a01 - a11) * (b01 + b11)
                U[1][c] = 1; U[3][c] = -1
                V[v_b01][c] = 1; V[v_b11][c] = 1
                W[w_c01][c] = 1; W[w_c11][c] = -1
            elif s == 6:
                # M7 = (a00 + a01) * (b10 - b11)
                U[0][c] = 1; U[1][c] = 1
                V[v_b10][c] = 1; V[v_b11][c] = -1
                W[w_c00][c] = 1; W[w_c10][c] = 1

    return {"U": U, "V": V, "W": W}
