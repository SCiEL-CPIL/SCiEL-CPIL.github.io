import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6 groups of 2 columns (Strassen) + 1 group of 1 column (Schoolbook)
    num_groups = 6
    rank = num_groups * 7 + 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    # Process 6 groups of 2 columns
    for g in range(1, num_groups + 1):
        # Columns 2g-1 and 2g
        k1 = 2 * g - 1
        k2 = 2 * g
        
        # V indices for B_{1, k1}, B_{1, k2}, B_{2, k1}, B_{2, k2}
        # V index = j * p + k
        v_11 = 1 * p + k1
        v_12 = 1 * p + k2
        v_21 = 2 * p + k1
        v_22 = 2 * p + k2
        
        # W indices for C_{1, k1}, C_{1, k2}, C_{2, k1}, C_{2, k2}
        # W index = k * n + i
        w_11 = k1 * n + 1
        w_12 = k2 * n + 1
        w_21 = k1 * n + 2
        w_22 = k2 * n + 2
        
        # U indices for A_{11}, A_{12}, A_{21}, A_{22}
        # U index = i * m + j
        u_11 = 1 * m + 1
        u_12 = 1 * m + 2
        u_21 = 2 * m + 1
        u_22 = 2 * m + 2
        
        # M1
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_11, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # M2
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_21, col] += 1
        W[w_22, col] += -1
        col += 1
        
        # M3
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_12, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # M4
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_11, col] += 1
        W[w_21, col] += 1
        col += 1
        
        # M5
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_11, col] += -1
        W[w_12, col] += 1
        col += 1
        
        # M6
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_22, col] += 1
        col += 1
        
        # M7
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_11, col] += 1
        col += 1

    # Process last group (column 13)
    k1 = 13
    v_11 = 1 * p + k1
    v_21 = 2 * p + k1
    w_11 = k1 * n + 1
    w_21 = k1 * n + 2
    
    # M1: A11 * B1,13
    U[u_11, col] += 1
    V[v_11, col] += 1
    W[w_11, col] += 1
    col += 1
    
    # M2: A12 * B2,13
    U[u_12, col] += 1
    V[v_21, col] += 1
    W[w_11, col] += 1
    col += 1
    
    # M3: A21 * B1,13
    U[u_21, col] += 1
    V[v_11, col] += 1
    W[w_21, col] += 1
    col += 1
    
    # M4: A22 * B2,13
    U[u_22, col] += 1
    V[v_21, col] += 1
    W[w_21, col] += 1
    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
