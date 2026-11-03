"""Program for 4x4x9 matrix multiplication rank decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    """
    Construct a rank-114 tensor decomposition for 4x4x9 matrix multiplication.
    
    Strategy:
    - Partition the 4x9 target into two 4x4 and one 4x1 sub-problems.
    - Use the rank-49 decomposition for 4x4x4 (derived from Bini's rank-7 2x2x2 tensor).
    - Use the rank-16 schoolbook decomposition for 4x4x1.
    - Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Bini's rank-7 decomposition for 2x2x2 matrix multiplication
    # Components as flattened 2x2 matrices (length 4)
    # Indices: 00, 01, 10, 11
    bini_u = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 1, 0]
    ]
    bini_v = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1]
    ]
    bini_w = [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1]
    ]
    
    # Generate rank-49 decomposition for 4x4x4 using Kronecker products
    u_4x4 = []
    v_4x4 = []
    w_4x4 = []
    for i in range(7):
        for j in range(7):
            u_4x4.append(np.kron(bini_u[i], bini_u[j]))
            v_4x4.append(np.kron(bini_v[i], bini_v[j]))
            w_4x4.append(np.kron(bini_w[i], bini_w[j]))
            
    rank_4x4 = 49
    rank_4x1 = 16
    total_rank = rank_4x4 * 2 + rank_4x1
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # Block 1: 4x4x4 (cols 0-3 of B)
    for r in range(rank_4x4):
        U[:, r] = u_4x4[r]
        V[0:16, r] = v_4x4[r]
        W[0:16, r] = w_4x4[r]
        
    # Block 2: 4x4x4 (cols 4-7 of B)
    for r in range(rank_4x4):
        U[:, rank_4x4 + r] = u_4x4[r]
        V[16:32, rank_4x4 + r] = v_4x4[r]
        W[16:32, rank_4x4 + r] = w_4x4[r]
        
    # Block 3: 4x4x1 (col 8 of B)
    # Schoolbook: rank 16
    for i in range(4):
        for j in range(4):
            col_idx = 2 * rank_4x4 + i * 4 + j
            U[i * 4 + j, col_idx] = 1
            V[32 + j, col_idx] = 1
            W[32 + i, col_idx] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
