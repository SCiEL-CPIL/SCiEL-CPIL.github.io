"""Matrix multiplication 2x5x6 - rank-minimal decomposition via block partitioning."""
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _get_rank_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.array([
        [1,  0,  0,  0,  0,  0,  0],  # i=0, m=0
        [0,  1,  0,  0,  0,  0,  0],  # i=0, m=1
        [0,  0,  1,  0,  1,  0,  0],  # i=1, m=0
        [0,  0,  1,  0,  0,  1,  0],  # i=1, m=1
    ], dtype=float)
    V = np.array([
        [1,  1,  0,  0,  0,  0,  0],  # m=0, k=0
        [0,  0,  1,  1,  0,  0,  0],  # m=0, k=1
        [1, -1,  0,  0,  0,  0,  0],  # m=1, k=0
        [0,  0,  1, -1,  0,  0,  0],  # m=1, k=1
    ], dtype=float)
    W = np.array([
        [0,  0,  1,  0,  0,  0,  0],  # k=0, i=0
        [1,  1,  0,  0,  0,  0,  0],  # k=0, i=1
        [0,  0,  0,  0,  1,  1,  1],  # k=1, i=0
        [0,  0,  0,  0,  0,  1,  0],  # k=1, i=1
    ], dtype=float)
    return U, V, W

def _get_rank_2x2x3():
    """Returns U, V, W for 2x2x3 (rank 6)."""
    # Schoolbook: 3 columns, each 2x2x1 (rank 2). Total 6.
    U = np.zeros((4, 6))
    V = np.zeros((6, 6))
    W = np.zeros((6, 6))
    
    # k=0
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # i=0, j=0, k=0
    U[1, 1] = 1; V[1, 1] = 1; W[1, 1] = 1 # i=0, j=1, k=0
    U[2, 2] = 1; V[2, 2] = 1; W[2, 2] = 1 # i=1, j=0, k=0
    U[3, 3] = 1; V[3, 3] = 1; W[3, 3] = 1 # i=1, j=1, k=0
    
    # k=1
    U[0, 4] = 1; V[4, 4] = 1; W[4, 4] = 1 # i=0, j=0, k=1
    U[1, 5] = 1; V[5, 5] = 1; W[5, 5] = 1 # i=0, j=1, k=1
    # Wait, W index is k*n + i.
    # For k=1, n=2:
    # i=0 -> index 2
    # i=1 -> index 3
    # My W array above has shape (6, 6). Rows are k*n+i.
    # k=0: i=0->0, i=1->1
    # k=1: i=0->2, i=1->3
    # k=2: i=0->4, i=1->5
    
    # Let's re-fill carefully.
    U = np.zeros((4, 6))
    V = np.zeros((6, 6))
    W = np.zeros((6, 6))
    
    col = 0
    for k in range(3):
        for i in range(2):
            for j in range(2):
                U[i*2 + j, col] = 1
                V[j*3 + k, col] = 1
                W[k*2 + i, col] = 1
                col += 1
    return U, V, W

def _get_rank_2x3x3():
    """Returns U, V, W for 2x3x3 (rank 12)."""
    # Split middle dim 3 = 2 + 1.
    # Part 1: 2x2x3 (rank 6)
    # Part 2: 2x1x3 (rank 6)
    U1, V1, W1 = _get_rank_2x2x3()
    
    U2 = np.zeros((2, 6)) # i=0,1; j=2 (only one col in A)
    V2 = np.zeros((3, 6)) # j=2; k=0,1,2
    W2 = np.zeros((6, 6)) # k=0,1,2; i=0,1
    
    col = 0
    for k in range(3):
        for i in range(2):
            U2[i, col] = 1
            V2[2*3 + k, col] = 1 # j=2, k=k -> index 2*3+k
            W2[k*2 + i, col] = 1
            col += 1
            
    U = np.zeros((6, 12)) # 2*3 = 6 rows
    V = np.zeros((9, 12)) # 3*3 = 9 rows
    W = np.zeros((6, 12)) # 3*2 = 6 rows
    
    # Fill Part 1 (j=0,1)
    for i in range(2):
        for j in range(2):
            U[i*3 + j, :6] = U1[i*2 + j, :]
    for j in range(2):
        for k in range(3):
            V[j*3 + k, :6] = V1[j*3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k*2 + i, :6] = W1[k*2 + i, :]
            
    # Fill Part 2 (j=2)
    for i in range(2):
        U[i*3 + 2, 6:] = U2[i, :]
    for k in range(3):
        V[2*3 + k, 6:] = V2[2, :] # V2 row index for j=2 is 0 in local, but we mapped it to 2*3+k in V2 construction? 
        # In V2 construction, V2 has shape (3,6). Row 0 is j=0? No, V2 was for j=2 only.
        # I defined V2 as (3,6) but only used row corresponding to j=2?
        # Let's re-define V2 to be just the entries for j=2.
        # V2_local[k] corresponds to V[j=2, k].
        # In the code above: V2[2*3 + k, col] = 1. This is wrong. V2 should be small.
        
    # Let's restart _get_rank_2x3x3 with cleaner logic.
    U = np.zeros((6, 12))
    V = np.zeros((9, 12))
    W = np.zeros((6, 12))
    
    # Part 1: 2x2x3 using U1, V1, W1
    U1, V1, W1 = _get_rank_2x2x3()
    
    # Map Part 1 to global indices
    # U1 rows: i in 0..1, j in 0..1 -> global i*3 + j
    # V1 rows: j in 0..1, k in 0..2 -> global j*3 + k
    # W1 rows: k in 0..2, i in 0..1 -> global k*2 + i
    for i in range(2):
        for j in range(2):
            U[i*3 + j, :6] = U1[i*2 + j, :]
    for j in range(2):
        for k in range(3):
            V[j*3 + k, :6] = V1[j*3 + k, :]
    for k in range(3):
        for i in range(2):
            W[k*2 + i, :6] = W1[k*2 + i, :]
            
    # Part 2: 2x1x3 (j=2)
    # U2: i in 0..1, j=2. Local index i. Global index i*3 + 2.
    # V2: j=2, k in 0..2. Local index k. Global index 2*3 + k.
    # W2: k in 0..2, i in 0..1. Local index k*2+i. Global index k*2+i.
    
    col = 6
    for k in range(3):
        for i in range(2):
            U[i*3 + 2, col] = 1
            V[2*3 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
            
    return U, V, W

def _get_rank_2x2x6():
    """Returns U, V, W for 2x2x6 (rank 12)."""
    # Split k=3+3.
    U1, V1, W1 = _get_rank_2x2x3()
    U2, V2, W2 = _get_rank_2x2x3()
    
    U = np.zeros((4, 12))
    V = np.zeros((12, 12))
    W = np.zeros((12, 12))
    
    for i in range(2):
        for j in range(2):
            U[i*2 + j, :6] = U1[i*2 + j, :]
            U[i*2 + j, 6:] = U2[i*2 + j, :]
            
    for j in range(2):
        for k in range(3):
            V[j*6 + k, :6] = V1[j*3 + k, :]
            V[j*6 + (3+k), 6:] = V2[j*3 + k, :]
            
    for k in range(3):
        for i in range(2):
            W[k*2 + i, :6] = W1[k*2 + i, :]
            W[(3+k)*2 + i, 6:] = W2[k*2 + i, :]
            
    return U, V, W

def _get_rank_2x3x6():
    """Returns U, V, W for 2x3x6 (rank 24)."""
    # Split k=3+3.
    U1, V1, W1 = _get_rank_2x3x3()
    U2, V2, W2 = _get_rank_2x3x3()
    
    U = np.zeros((6, 24))
    V = np.zeros((18, 24))
    W = np.zeros((12, 24))
    
    for i in range(2):
        for j in range(3):
            U[i*3 + j, :12] = U1[i*3 + j, :]
            U[i*3 + j, 12:] = U2[i*3 + j, :]
            
    for j in range(3):
        for k in range(3):
            V[j*6 + k, :12] = V1[j*3 + k, :]
            V[j*6 + (3+k), 12:] = V2[j*3 + k, :]
            
    for k in range(3):
        for i in range(2):
            W[k*2 + i, :12] = W1[k*2 + i, :]
            W[(3+k)*2 + i, 12:] = W2[k*2 + i, :]
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # Split m=5 into 2+3
    # Part 1: 2x2x6
    U1, V1, W1 = _get_rank_2x2x6()
    # Part 2: 2x3x6
    U2, V2, W2 = _get_rank_2x3x6()
    
    rank = 12 + 24 # 36
    
    U = np.zeros((10, rank)) # 2*5
    V = np.zeros((30, rank)) # 5*6
    W = np.zeros((12, rank)) # 6*2
    
    # Part 1: j=0,1
    for i in range(2):
        for j in range(2):
            U[i*5 + j, :12] = U1[i*2 + j, :]
    for j in range(2):
        for k in range(6):
            V[j*6 + k, :12] = V1[j*6 + k, :]
    for k in range(6):
        for i in range(2):
            W[k*2 + i, :12] = W1[k*2 + i, :]
            
    # Part 2: j=2,3,4
    for i in range(2):
        for j in range(3):
            U[i*5 + (2+j), 12:] = U2[i*3 + j, :]
    for j in range(3):
        for k in range(6):
            V[(2+j)*6 + k, 12:] = V2[j*6 + k, :]
    for k in range(6):
        for i in range(2):
            W[k*2 + i, 12:] = W2[k*2 + i, :]
            
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
