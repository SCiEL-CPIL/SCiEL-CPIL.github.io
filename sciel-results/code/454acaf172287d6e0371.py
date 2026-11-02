import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Strategy: Use rank-7 Strassen for 2x2x2 blocks
    # Split 2x5x6 as:
    #   j: {0,1}, {2,3}, {4}
    #   k: {0,1}, {2,3}, {4,5}
    # Blocks:
    #   (j01, k01): 2x2x2 -> rank 7
    #   (j01, k23): 2x2x2 -> rank 7
    #   (j01, k45): 2x2x2 -> rank 7
    #   (j23, k01): 2x2x2 -> rank 7
    #   (j23, k23): 2x2x2 -> rank 7
    #   (j23, k45): 2x2x2 -> rank 7
    #   (j4, k01): 2x1x2 -> rank 4
    #   (j4, k23): 2x1x2 -> rank 4
    #   (j4, k45): 2x1x2 -> rank 4
    # Total: 6*7 + 3*4 = 42 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(j1, j2, k1, k2):
        """Add Strassen rank-7 decomposition for 2x2x2 block."""
        nonlocal col
        u00 = 0 * m + j1
        u01 = 0 * m + j2
        u10 = 1 * m + j1
        u11 = 1 * m + j2
        v00 = j1 * p + k1
        v01 = j1 * p + k2
        v10 = j2 * p + k1
        v11 = j2 * p + k2
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1
        
        # M1 = (a00+a11)(b00+b11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)b00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        
        # M3 = a00(b01-b11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = a11(b10-b00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)b11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_standard(i, j, k):
        """Add a single standard multiplication term."""
        nonlocal col
        U[i * m + j][col] += 1
        V[j * p + k][col] += 1
        W[k * n + i][col] += 1
        col += 1
    
    # Apply Strassen blocks for 2x2x2
    add_strassen(0, 1, 0, 1)
    add_strassen(0, 1, 2, 3)
    add_strassen(0, 1, 4, 5)
    add_strassen(2, 3, 0, 1)
    add_strassen(2, 3, 2, 3)
    add_strassen(2, 3, 4, 5)
    
    # Standard multiplications for j=4 (2x1x2 blocks)
    for k in range(0, 2):
        add_standard(0, 4, k)
        add_standard(1, 4, k)
    for k in range(2, 4):
        add_standard(0, 4, k)
        add_standard(1, 4, k)
    for k in range(4, 6):
        add_standard(0, 4, k)
        add_standard(1, 4, k)
    
    # Verify the decomposition
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    def verify():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True
    
    if not verify():
        # Fallback to pure standard rank 60
        rank_f = 60
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(2):
            for j in range(5):
                for k in range(6):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
