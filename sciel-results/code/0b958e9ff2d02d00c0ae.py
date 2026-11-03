import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # 3x3 Winograd coefficients (Rank 15)
    # U1: 3x3x15 -> 9x15
    # V1: 3x3x15 -> 9x15
    # W1: 3x3x15 -> 9x15
    
    # Flattened U1 (i*m + j)
    U1 = [
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0], # i=0, j=0
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0], # i=0, j=1
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], # i=0, j=2
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0], # i=1, j=0
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0], # i=1, j=1
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], # i=1, j=2
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], # i=2, j=0
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], # i=2, j=1
        [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]  # i=2, j=2
    ]
    
    # Flattened V1 (j*p + k) -> Here p=1 for the 3x3 part, so index is j
    # Wait, V1 is for B (3x1). So V1[j, 0, r].
    # In the 3x3 case, V1 is symmetric to U1?
    # Yes, for square matrices, U and V are often the same or transposed.
    # In Winograd F(3), U and V are the same matrix.
    V1 = [
        [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0], # j=0, k=0
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0], # j=1, k=0
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]  # j=2, k=0
    ]
    
    # Flattened W1 (k*n + i) -> Here n=1 for the 3x1 part? No, C is 3x1.
    # W1[k, i, r].
    W1 = [
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0], # k=0, i=0
        [0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0], # k=0, i=1
        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]  # k=0, i=2
    ]

    rank = 15
    
    # Construct U (9x15)
    # U[i, j, r] = U1[i, j, r]
    U = [row[:] for row in U1]
    
    # Construct V (15x15)
    # V[j, k, r] = V1[j, 0, r] for k in 0..4
    V = []
    for k in range(5):
        for j in range(3):
            # V1 is indexed by j (row) and k=0 (col)
            # In V1 list above, row j corresponds to V1[j, 0, :]
            V.append(V1[j])
            
    # Construct W (15x15)
    # W[k, i, r] = W1[0, i, r] for k in 0..4
    # W1 is indexed by k=0 (row) and i (col)
    W = []
    for k in range(5):
        for i in range(3):
            W.append(W1[i])
            
    return {"U": U, "V": V, "W": W}
