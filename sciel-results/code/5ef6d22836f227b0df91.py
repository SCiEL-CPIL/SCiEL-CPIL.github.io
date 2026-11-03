import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split p=6 into two blocks of size 3
    p1 = 3
    p2 = 3
    
    # Helper to build schoolbook for n x m x p_block
    def build_schoolbook(n, m, p_block):
        rank = n * m * p_block
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p_block, rank))
        W = np.zeros((p_block * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    U[i * m + j, col] = 1
                    V[j * p_block + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank

    # Block 1: k in [0, 3)
    U1, V1, W1, r1 = build_schoolbook(n, m, p1)
    # Block 2: k in [3, 6)
    U2, V2, W2, r2 = build_schoolbook(n, m, p2)
    
    # Total rank is r1 + r2
    rank = r1 + r2
    
    # U is (n*m, rank)
    # U1 is (n*m, r1), U2 is (n*m, r2)
    U = np.hstack([U1, U2])
    
    # V is (m*p, rank)
    V_full = np.zeros((m * p, rank))
    # Place V1 (m*p1, r1) into V_full
    # V1 row j*p1 + k_local corresponds to V_full row j*p + k_local
    for j in range(m):
        for k_local in range(p1):
            local_row = j * p1 + k_local
            global_row = j * p + k_local
            V_full[global_row, :r1] = V1[local_row, :]
            
    # Place V2 (m*p2, r2) into V_full
    # V2 row j*p2 + k_local corresponds to V_full row j*p + (p1 + k_local)
    for j in range(m):
        for k_local in range(p2):
            local_row = j * p2 + k_local
            global_row = j * p + (p1 + k_local)
            V_full[global_row, r1:] = V2[local_row, :]

    # W is (p*n, rank)
    W_full = np.zeros((p * n, rank))
    # Place W1 (p1*n, r1) into W_full
    # W1 row k_local*n + i corresponds to W_full row k_local*n + i
    W_full[:p1*n, :r1] = W1
    
    # Place W2 (p2*n, r2) into W_full
    # W2 row k_local*n + i corresponds to W_full row (p1 + k_local)*n + i
    W_full[p1*n:, r1:] = W2

    return {"U": U.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
