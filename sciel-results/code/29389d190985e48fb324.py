import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # The tensor is a direct sum of p instances of M_{n,m,1}.
    # Rank of M_{2,2,1} is 4.
    # Total rank = p * 4 = 9 * 4 = 36.
    
    rank = n * m * p
    # Initialize U, V, W as lists of lists to match expected output format
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # We can construct this efficiently by recognizing the block structure.
    # For each k in 0..p-1, we have a 2x2x1 multiplication.
    # The 4 terms for a single k are:
    # 1. A[0,0]*B[0,k] -> C[0,k]
    # 2. A[0,1]*B[1,k] -> C[0,k]
    # 3. A[1,0]*B[0,k] -> C[1,k]
    # 4. A[1,1]*B[1,k] -> C[1,k]
    
    # Indices:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We fill the columns corresponding to each k and each (i,j) pair.
    # Column index c = k * 4 + idx, where idx is 0..3 for the 4 terms.
    
    # Term 0: i=0, j=0
    # Term 1: i=0, j=1
    # Term 2: i=1, j=0
    # Term 3: i=1, j=1
    
    for k in range(p):
        base_col = k * 4
        
        # i=0, j=0
        U[0 * m + 0, base_col + 0] = 1
        V[0 * p + k, base_col + 0] = 1
        W[k * n + 0, base_col + 0] = 1
        
        # i=0, j=1
        U[0 * m + 1, base_col + 1] = 1
        V[1 * p + k, base_col + 1] = 1
        W[k * n + 0, base_col + 1] = 1
        
        # i=1, j=0
        U[1 * m + 0, base_col + 2] = 1
        V[0 * p + k, base_col + 2] = 1
        W[k * n + 1, base_col + 2] = 1
        
        # i=1, j=1
        U[1 * m + 1, base_col + 3] = 1
        V[1 * p + k, base_col + 3] = 1
        W[k * n + 1, base_col + 3] = 1

    # Convert to list of lists as required by the interface description 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The example returns lists of lists.
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
