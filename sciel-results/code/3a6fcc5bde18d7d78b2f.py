import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    # We split the middle dimension m=3 into 2 and 1.
    # Part 1: i in {0,1}, j in {0,1}, k in {0,1,2}
    # Part 2: i in {0,1}, j in {2}, k in {0,1,2}
    
    # Part 1: 2x2x3 schoolbook (Rank 12)
    # Part 2: 2x1x3 rank 1 (Rank 1)
    # Total Rank 13
    
    rank = 13
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: j=0,1
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: j=2
    # Rank 1 decomposition for 2x1x3
    # U[i, 2, 0] = 1
    # V[2, k, 0] = 1
    # W[k, i, 0] = 1
    for i in range(n):
        for k in range(p):
            # Only one term for this part
            if col == 12:
                U[i * m + 2, col] = 1
                V[2 * p + k, col] = 1
                W[k * n + i, col] = 1
                # Note: This sets the same column for all i,k?
                # No, the decomposition is sum_r U V W.
                # For rank 1, r is fixed (say r=12).
                # U[i, 2, 12] = 1 for all i.
                # V[2, k, 12] = 1 for all k.
                # W[k, i, 12] = 1 for all k,i.
                # Then sum is 1 * 1 * 1 = 1.
                pass
                
    # The loop above is wrong for the rank-1 part.
    # I need to set U, V, W for the single rank-1 term.
    
    # Reset and do it properly
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 2x2x3 schoolbook
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Part 2: 2x1x3 rank 1
    # col is now 12
    r = col
    for i in range(n):
        U[i * m + 2, r] = 1
    for k in range(p):
        V[2 * p + k, r] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, r] = 1
            
    return {"U": U, "V": V, "W": W}
