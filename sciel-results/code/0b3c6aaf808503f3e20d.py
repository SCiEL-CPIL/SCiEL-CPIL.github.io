"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 140

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Decomposition: split B (4x9) into B1(4x4), B2(4x4), B3(4x1)
    # C = A*B = [A*B1 | A*B2 | A*B3]
    # Using schoolbook for each part: rank = 64 + 64 + 4 = 132
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Part 1: A(4x4) * B1(4x4) -> C1(4x4), output columns 0-3
    for i in range(n):
        for j in range(m):
            for k in range(4):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    # Part 2: A(4x4) * B2(4x4) -> C2(4x4), output columns 4-7
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    # Part 3: A(4x4) * B3(4x1) -> C3(4x1), output column 8
    for i in range(n):
        for j in range(m):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
    
    rank = len(U_cols)
    
    U = [[U_cols[c][r] for c in range(rank)] for r in range(n * m)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(m * p)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
