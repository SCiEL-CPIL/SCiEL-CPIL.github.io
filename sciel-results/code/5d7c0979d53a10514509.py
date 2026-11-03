import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank
    rank = 31
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:7]
    # Schoolbook for 2x2 by 2x7
    # A indices: i in 0..1, j in 0..1
    # B indices: j in 0..1, k in 0..6
    # C indices: i in 0..1, k in 0..6
    for i in range(2):
        for j in range(2):
            for k in range(7):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Block 2: A[0:2, 2] * B[2, 0:7]
    # Rank 1
    # A indices: i in 0..1, j=2
    # B indices: j=2, k in 0..6
    # C indices: i in 0..1, k in 0..6
    for i in range(2):
        for k in range(7):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
    col += 1
    
    # Block 3: A[2, 0:2] * B[0:2, 0:7]
    # Rank 1
    # A indices: i=2, j in 0..1
    # B indices: j in 0..1, k in 0..6
    # C indices: i=2, k in 0..6
    for k in range(7):
        U[2 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 2][col] = 1
        U[2 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 2][col] = 1 # Wait, W is shared?
        
    # Correction: The rank 1 decomposition for A[2, 0:2] * B[0:2, 0:7]
    # T_{2, j, k} = 1 for j=0,1.
    # U_{2, r} V_{j, r} W_{k, r} = 1.
    # If we use one column r:
    # U_{2, r} = 1
    # V_{0, r} = 1, V_{1, r} = 1
    # W_{k, r} = 1
    # Then for j=0: 1*1*1 = 1.
    # For j=1: 1*1*1 = 1.
    # This works.
    # So we only need 1 column.
    col -= 1 # Undo the increment from the previous block if I messed up
    # Actually, let's just add 1 column.
    U[2 * m + 0][col] = 1
    U[2 * m + 1][col] = 1
    V[0 * p + 0][col] = 1 # Wait, V index is j*p+k.
    # V[0, k] and V[1, k] must be 1.
    # But V is a matrix.
    # V[j*p+k, r].
    # For j=0, k=0..6: V[0*p+k, r] = 1.
    # For j=1, k=0..6: V[1*p+k, r] = 1.
    for k in range(7):
        V[0 * p + k][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 2][col] = 1
    col += 1
    
    # Block 4: A[2, 2] * B[2, 0:7]
    # Rank 1
    U[2 * m + 2][col] = 1
    for k in range(7):
        V[2 * p + k][col] = 1
        W[k * n + 2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
