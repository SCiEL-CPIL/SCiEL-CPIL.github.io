import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Part 1: 4x4x4 using Rank 49 (Kron of Rank 7)
    # Strassen Rank 7 factors for 2x2
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Kronecker product for 4x4x4
    U44 = np.kron(U2, U2)
    V44 = np.kron(V2, V2)
    W44 = np.kron(W2, W2)
    
    # Note: The row order of np.kron might need permutation to match 4*i+j
    # For U, rows are 4*i + j.
    # np.kron(U2, U2) rows are 4*i2 + j2 where i2 is row in first U2, j2 in second.
    # If i2 corresponds to "block" index and j2 to "inner" index, we need to check.
    # Standard Kronecker product A (m x n) and B (p x q) has rows (i, j) -> i*p + j.
    # Here i in 0..3, j in 0..3.
    # This matches 4*i + j.
    # So the row order is correct if the indices align.
    
    # Part 2: 4x4x3 using Schoolbook Rank 48
    rank48 = 4 * 4 * 3
    U43 = np.zeros((16, rank48))
    V43 = np.zeros((12, rank48))
    W43 = np.zeros((12, rank48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U43[4*i + j, col] = 1
                V43[4*j + k, col] = 1
                W43[4*k + i, col] = 1
                col += 1
                
    # Concatenate
    U = np.hstack([U44, U43])
    V = np.hstack([V44, V43])
    W = np.hstack([W44, W43])
    
    # Convert to list of lists for return
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
