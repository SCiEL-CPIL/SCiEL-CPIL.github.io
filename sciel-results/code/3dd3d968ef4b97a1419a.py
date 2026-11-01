"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = 2, 2, 13
    
    # Strategy: Use Strassen decomposition for 2x2 * 2x2 blocks
    # 13 columns = 6 blocks of 2 columns + 1 column
    # Each 2-column block uses rank-7 Strassen
    # Last single column uses rank-4 schoolbook
    # Total rank: 6*7 + 4 = 46
    
    # Strassen decomposition for 2x2 * 2x2:
    # P1 = (a00+a11)(b00+b11)
    # P2 = (a10+a11)b00
    # P3 = a00(b10-b11)
    # P4 = a11(b01-b00)
    # P5 = (a00+a01)b11
    # P6 = (a10-a00)(b00+b10)
    # P7 = (a01-a11)(b01+b11)
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P1 - P2 + P6
    # C10 = P2 + P4 - P3 + P5
    # C11 = P1 - P4 + P2 - P6
    
    # U indices: (i,k) -> i*2+k, so [0]=a00, [1]=a01, [2]=a10, [3]=a11
    # V indices: (k,j) -> k*2+j, so [0]=b00, [1]=b01, [2]=b10, [3]=b11
    # W indices: (j,i) -> j*2+i, so [0]=C00, [1]=C01, [2]=C10, [3]=C11
    
    # Strassen factor matrices (4x7 for U and W, 4x7 for V)
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # P1: a00+a11
        [0, 0, 0, 0, 1, 0, 1],    # P2: a11 (wait, let me recheck)
    ])
    
    # Let me be very careful:
    # P1 = (a00+a11)(b00+b11): U=[1,0,0,1], V=[1,0,0,1], W=[1,1,0,1]
    # P2 = (a10+a11)b00:         U=[0,0,1,1], V=[1,0,0,0], W=[0,-1,1,1]
    # P3 = a00(b10-b11):         U=[1,0,0,0], V=[0,0,1,-1], W=[0,1,1,0]
    # P4 = a11(b01-b00):         U=[0,0,0,1], V=[0,1,-1,0], W=[1,0,0,-1]
    # P5 = (a00+a01)b11:         U=[1,1,0,0], V=[0,0,0,1], W=[0,0,-1,0]
    # P6 = (a10-a00)(b00+b10):   U=[-1,0,1,0], V=[1,0,1,0], W=[0,1,0,-1]
    # P7 = (a01-a11)(b01+b11):   U=[0,1,0,-1], V=[0,1,0,1], W=[1,0,0,0]
    
    strassen_U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1],
    ])
    strassen_V = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, -1, 0, 1, 0],
        [1, 0, -1, 0, 1, 0, 1],
    ])
    strassen_W = np.array([
        [1, 0, 0, 1, 0, 0, 1],
        [1, -1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, -1, 0, 0],
        [1, 1, 0, -1, 0, -1, 0],
    ])
    
    # Verify Strassen decomposition numerically
    S_target = np.zeros((4, 4, 4), dtype=float)
    S_target[0, 0, 0] = 1  # a00 * b00 -> C00
    S_target[0, 1, 2] = 1  # a00 * b10 -> C10
    S_target[1, 2, 0] = 1  # a01 * b01 -> C01
    S_target[1, 3, 2] = 1  # a01 * b11 -> C11
    S_target[2, 0, 1] = 1  # a10 * b00 -> C01
    S_target[2, 1, 3] = 1  # a10 * b01 -> C11
    S_target[3, 2, 1] = 1  # a11 * b10 -> C01
    S_target[3, 3, 3] = 1  # a11 * b11 -> C11
    
    S_recon = np.zeros((4, 4, 4), dtype=float)
    for t in range(7):
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    S_recon[a, b, c] += strassen_U[a][t] * strassen_V[b][t] * strassen_W[c][t]
    
    if not np.allclose(S_recon, S_target, atol=1e-10):
        # Fallback to schoolbook if Strassen verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build full decomposition
    # 6 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    # 1 schoolbook block for column 12
    total_rank = 6 * 7 + 4  # = 46
    
    U_full = np.zeros((n * m, total_rank), dtype=float)  # (4, 46)
    V_full = np.zeros((m * p, total_rank), dtype=float)  # (26, 46)
    W_full = np.zeros((p * n, total_rank), dtype=float)  # (26, 46)
    
    col = 0
    # 6 Strassen blocks
    for block in range(6):
        j0 = 2 * block
        j1 = 2 * block + 1
        for t in range(7):
            # U: same for all blocks (operates on A entries)
            U_full[:, col] = strassen_U[:, t]
            # V: map local (k, j_local) to global (k, j)
            # local V index: k*2 + j_local where j_local 0->j0, 1->j1
            for k in range(2):
                for jl in range(2):
                    j_global = j0 + jl
                    v_local = k * 2 + jl
                    v_global = k * p + j_global
                    V_full[v_global, col] = strassen_V[v_local, t]
            # W: map local (j_local, i) to global (j, i)
            for jl in range(2):
                for i in range(2):
                    j_global = j0 + jl
                    w_local = jl * 2 + i
                    w_global = j_global * n + i
                    W_full[w_global, col] = strassen_W[w_local, t]
            col += 1
    
    # 1 schoolbook block for column 12
    j = 12
    for i in range(n):
        for k in range(m):
            U_full[i * m + k, col] = 1
            V_full[k * p + j, col] = 1
            W_full[j * n + i, col] = 1
            col += 1
    
    # Verify full decomposition
    T_target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T_target[i * m + k, k * p + j, j * n + i] = 1
    
    T_recon = np.zeros((n * m, m * p, p * n), dtype=float)
    for t in range(total_rank):
        for a in range(n * m):
            ua = U_full[a, t]
            if ua == 0:
                continue
            for b in range(m * p):
                ub = ua * V_full[b, t]
                if ub == 0:
                    continue
                for c in range(p * n):
                    T_recon[a, b, c] += ub * W_full[c, t]
    
    if not np.allclose(T_recon, T_target, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k][column] = 1
                    V[k * p + j][column] = 1
                    W[j * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to list of lists for output
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
