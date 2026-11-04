import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _schoolbook_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 9.
    U: (9, 9)
    V: (9, 9)
    W: (9, 9)
    """
    n, m, p = 3, 3, 3
    rank = 9
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

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = DIMENSIONS
    
    # Get the 3x3x3 decomposition
    U3, V3, W3 = _schoolbook_3x3x3()
    
    # We need to construct the 3x3x6 decomposition.
    # This is the direct sum of two 3x3x3 decompositions.
    # Part 1: B columns 0-2
    # Part 2: B columns 3-5
    
    rank_total = 18
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # Copy Part 1
    U[:, 0:9] = U3
    V[:, 0:9] = V3
    # For W, the indices are k*n + i. k is 0,1,2.
    # This matches the first 9 entries of W3 (which are for k=0,1,2).
    # W3 has shape (9, 9). Indices 0..8 correspond to k=0,1,2 and i=0,1,2.
    # Specifically, W3[k*n+i, col] is 1.
    # In the full W, the index is k*n + i.
    # For k=0,1,2, the indices are 0..8.
    # So W[:, 0:9] = W3 works directly.
    W[:, 0:9] = W3
    
    # Copy Part 2
    U[:, 9:18] = U3
    # For V, the indices are j*p + k.
    # In the full V, p=6.
    # Part 2 corresponds to B columns 3,4,5.
    # So for a given j and local k (0,1,2), the global k is 3+k.
    # The index in V is j*6 + (3+k).
    # In V3, the index is j*3 + k.
    # We need to map V3 indices to V indices.
    V2 = np.zeros((m * p, 9))
    for j in range(m):
        for k in range(3):
            # V3 index
            idx_v3 = j * 3 + k
            # V full index
            idx_v_full = j * 6 + (3 + k)
            V2[idx_v_full, :] = V3[idx_v3, :]
    V[:, 9:18] = V2
    
    # For W, the indices are k*n + i.
    # Part 2 corresponds to C columns 3,4,5.
    # So for a given global k (3,4,5) and i, the index is k*3 + i.
    # In W3, the index is local_k*3 + i, where local_k is 0,1,2.
    # So global_k = 3 + local_k.
    # Index in W full is (3+local_k)*3 + i.
    W2 = np.zeros((p * n, 9))
    for local_k in range(3):
        for i in range(n):
            # W3 index
            idx_w3 = local_k * 3 + i
            # W full index
            idx_w_full = (3 + local_k) * 3 + i
            W2[idx_w_full, :] = W3[idx_w3, :]
    W[:, 9:18] = W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
