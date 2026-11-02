import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Split p=6 into two blocks of size 3
    p1 = 3
    p2 = 3
    
    rank_total = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to build schoolbook for n x m x p_block
    def build_schoolbook(n, m, p_block, offset_k):
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
    U1, V1, W1, r1 = build_schoolbook(n, m, p1, 0)
    # Block 2: k in [3, 6)
    U2, V2, W2, r2 = build_schoolbook(n, m, p2, 0)
    
    # Concatenate U
    U = np.hstack([U1, U2])
    
    # Concatenate V
    # V1 is (m*p1, r1) = (9, 9)
    # V2 is (m*p2, r2) = (9, 9)
    # We need V to be (m*p, rank) = (18, 18)
    # V indices: j*p + k
    # For block 1, k is 0..2. V1 covers j*3 + k for k=0..2.
    # For block 2, k is 3..5. V2 covers j*3 + (k-3) for k=3..5.
    # We need to map V2 to the correct rows in the full V matrix.
    V_full = np.zeros((m * p, r1 + r2))
    V_full[:, :r1] = V1
    # V2 corresponds to k=0..2 in local indices, which are k=3..5 in global.
    # Row index in V_full is j*p + k_global.
    # Row index in V2 is j*p1 + k_local.
    # j*p1 + k_local = j*3 + k_local.
    # Global row = j*6 + (3 + k_local).
    # We need to place V2[local_row, col] into V_full[global_row, r1+col].
    for j in range(m):
        for k_local in range(p2):
            local_row = j * p2 + k_local
            global_row = j * p + (p1 + k_local)
            V_full[global_row, r1:] = V2[local_row, :]
            
    # Concatenate W
    # W1 is (p1*n, r1) = (6, 9)
    # W2 is (p2*n, r2) = (6, 9)
    # W indices: k*n + i
    # W1 covers k=0..2.
    # W2 covers k=3..5.
    W_full = np.zeros((p * n, r1 + r2))
    W_full[:, :r1] = W1
    for k_local in range(p2):
        local_row = k_local * n
        global_row = (p1 + k_local) * n
        W_full[global_row:global_row+n, r1:] = W2[local_row:local_row+n, :]
        
    rank = r1 + r2
    return {"U": U.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
