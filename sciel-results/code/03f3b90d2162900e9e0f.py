"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the n x m by m x p multiplication."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    rank = U_np.shape[1]
    max_err = 0.0
    for a in range(n * m):
        i = a // m
        j = a % m
        for b in range(m * p):
            j_b = b // p
            k = b % p
            for c in range(p * n):
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (j_b == j and k_c == k and i_c == i) else 0.0
                val = sum(U_np[a, cc] * V_np[b, cc] * W_np[c, cc] for cc in range(rank))
                max_err = max(max_err, abs(val - expected))
    return max_err

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
    # Block decomposition:
    # Block 1: output columns 0-2 (3x3x3 schoolbook, rank 27)
    # Block 2: output columns 3-5 (3x3x3 schoolbook, rank 27)  
    # Block 3: output columns 6-7 (3x3x2 schoolbook, rank 6)
    # Total rank: 60
    
    rank = 60
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: k = 0, 1, 2
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    # Block 2: k = 3, 4, 5
    for i in range(n):
        for j in range(m):
            for k in range(3, 6):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    # Block 3: k = 6, 7
    for i in range(n):
        for j in range(m):
            for k in range(6, 8):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    # Verify if time permits
    if time.time() < DEADLINE - 5:
        err = _verify_decomposition(U, V, W, n, m, p)
        if err > 1e-9:
            # Fallback to full schoolbook
            rank = n * m * p
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][c] = 1.0
                        V[j * p + k][c] = 1.0
                        W[k * n + i][c] = 1.0
                        c += 1
    
    U_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in U]
    V_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in V]
    W_out = [[int(round(x)) if abs(x - round(x)) < 1e-9 else round(x, 6) for x in row] for row in W]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
