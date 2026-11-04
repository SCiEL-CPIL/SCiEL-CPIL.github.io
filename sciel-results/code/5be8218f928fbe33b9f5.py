import numpy as np
import time

# The deadline guard is kept for safety, though this specific construction is deterministic and fast.
_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor.
    This is derived from the Kronecker product of the rank-7 Strassen decomposition
    for 2x2 matrix multiplication.
    
    Returns:
        dict: A dictionary with keys 'U', 'V', 'W' containing the factor matrices.
              U has shape (16, 49), V has shape (16, 49), W has shape (16, 49).
    """
    if not _time_ok():
        # Fallback to a trivial valid decomposition if time is critically low?
        # Actually, if we are past deadline, we should return something valid immediately.
        # But this function is fast. We can just proceed.
        pass

    n, m, p = 4, 4, 4
    rank = 49
    
    # Define the rank-7 Strassen decomposition for 2x2 matrices.
    # U2, V2, W2 are 4x7 matrices.
    # Indices: 0..3 for rows, 0..6 for columns (rank).
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 1, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ]
    
    W2 = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 1, 0, 1, 0]
    ]
    
    # Convert to numpy arrays for easier manipulation, or keep as lists.
    # Lists are fine for small sizes.
    
    # Pre-compute sparse representations for efficiency in construction
    def to_sparse(matrix):
        sparse = []
        for i in range(len(matrix)):
            row_sparse = []
            for j in range(len(matrix[0])):
                if matrix[i][j] != 0:
                    row_sparse.append((j, matrix[i][j]))
            sparse.append(row_sparse)
        return sparse

    U2_sp = to_sparse(U2)
    V2_sp = to_sparse(V2)
    W2_sp = to_sparse(W2)

    # Initialize output matrices
    # U shape: (n*m, rank) = (16, 49)
    # V shape: (m*p, rank) = (16, 49)
    # W shape: (p*n, rank) = (16, 49)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Construct U
    # U[i*m + j, c1*7 + c2] = U2[i1*2 + j1, c1] * U2[i2*2 + j2, c2]
    for i in range(n):
        for j in range(m):
            i1 = i // 2
            i2 = i % 2
            j1 = j // 2
            j2 = j % 2
            
            row_idx = i * m + j
            u2_row1_idx = i1 * 2 + j1
            u2_row2_idx = i2 * 2 + j2
            
            for c1, v1 in U2_sp[u2_row1_idx]:
                for c2, v2 in U2_sp[u2_row2_idx]:
                    col_idx = c1 * 7 + c2
                    U[row_idx][col_idx] = v1 * v2

    # Construct V
    # V[j*p + k, c1*7 + c2] = V2[j1*2 + k1, c1] * V2[j2*2 + k2, c2]
    for j in range(m):
        for k in range(p):
            j1 = j // 2
            j2 = j % 2
            k1 = k // 2
            k2 = k % 2
            
            row_idx = j * p + k
            v2_row1_idx = j1 * 2 + k1
            v2_row2_idx = j2 * 2 + k2
            
            for c1, v1 in V2_sp[v2_row1_idx]:
                for c2, v2 in V2_sp[v2_row2_idx]:
                    col_idx = c1 * 7 + c2
                    V[row_idx][col_idx] = v1 * v2

    # Construct W
    # W[k*n + i, c1*7 + c2] = W2[k1*2 + i1, c1] * W2[k2*2 + i2, c2]
    for k in range(p):
        for i in range(n):
            k1 = k // 2
            k2 = k % 2
            i1 = i // 2
            i2 = i % 2
            
            row_idx = k * n + i
            w2_row1_idx = k1 * 2 + i1
            w2_row2_idx = k2 * 2 + i2
            
            for c1, v1 in W2_sp[w2_row1_idx]:
                for c2, v2 in W2_sp[w2_row2_idx]:
                    col_idx = c1 * 7 + c2
                    W[row_idx][col_idx] = v1 * v2

    return {"U": U, "V": V, "W": W}
