import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition for the full tensor is rank 63.
    # We construct a block decomposition.
    # Split p=7 into 3, 3, 1.
    
    # Helper to create schoolbook decomposition for M_{n,m,p}
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Block 1: M_{3,3,3} for k=0,1,2
    U1, V1, W1 = schoolbook(n, m, 3)
    # Block 2: M_{3,3,3} for k=3,4,5
    U2, V2, W2 = schoolbook(n, m, 3)
    # Block 3: M_{3,3,1} for k=6
    U3, V3, W3 = schoolbook(n, m, 1)
    
    rank1 = U1.shape[1]
    rank2 = U2.shape[1]
    rank3 = U3.shape[1]
    total_rank = rank1 + rank2 + rank3
    
    # Initialize total matrices
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    # Fill Block 1
    U_total[:, 0:rank1] = U1
    # V1 has shape (m*3, rank1). V_total has shape (m*7, total_rank).
    # Rows 0-8 of V_total correspond to k=0,1,2.
    V_total[0:m*3, 0:rank1] = V1
    # W1 has shape (3*n, rank1). W_total has shape (7*n, total_rank).
    # Rows 0-8 of W_total correspond to k=0,1,2.
    W_total[0:3*n, 0:rank1] = W1
    
    # Fill Block 2
    U_total[:, rank1:rank1+rank2] = U2
    # V2 has shape (m*3, rank2).
    # Rows 9-17 of V_total correspond to k=3,4,5.
    V_total[m*3:m*6, rank1:rank1+rank2] = V2
    # W2 has shape (3*n, rank2).
    # Rows 9-17 of W_total correspond to k=3,4,5.
    W_total[3*n:6*n, rank1:rank1+rank2] = W2
    
    # Fill Block 3
    U_total[:, rank1+rank2:total_rank] = U3
    # V3 has shape (m*1, rank3).
    # Rows 18-20 of V_total correspond to k=6.
    V_total[m*6:m*7, rank1+rank2:total_rank] = V3
    # W3 has shape (1*n, rank3).
    # Rows 18-20 of W_total correspond to k=6.
    W_total[6*n:7*n, rank1+rank2:total_rank] = W3
    
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
