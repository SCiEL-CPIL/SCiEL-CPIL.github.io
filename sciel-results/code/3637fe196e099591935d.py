import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    # The tensor for M_{2,2,6} is the direct sum of three M_{2,2,2} tensors.
    # The rank of M_{2,2,2} is 7 (Strassen).
    # Therefore, the minimum rank for M_{2,2,6} is 3 * 7 = 21.
    # We construct the rank 21 decomposition directly.
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen decomposition for 2x2 matrices
    # U, V, W are 4x1 matrices (flattened 2x2)
    # A, B, C are 2x2 matrices
    # C = A * B
    # U_s, V_s, W_s are vectors of length 4 (row-major: [00, 01, 10, 11])
    strassen = [
        # U, V, W
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]},
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]},
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]},
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]},
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]},
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]},
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]},
    ]
    
    # Initialize factor matrices
    # U: (n*m, rank) = (4, 21)
    # V: (m*p, rank) = (12, 21)
    # W: (p*n, rank) = (12, 21)
    # Note: W index is k*n + i. k in 0..5, i in 0..1.
    # So W row index c = k*2 + i.
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # The 2x2x6 multiplication consists of 3 independent 2x2x2 multiplications
    # corresponding to k blocks: {0,1}, {2,3}, {4,5}
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            # Fill U (shared across blocks, but since they are independent,
            # we just repeat the U pattern for each block's columns)
            # Actually, U is the same for all blocks because A is the same.
            # The columns of U correspond to the rank-1 terms.
            # For block b, the rank-1 terms are at columns b*7 .. b*7+6.
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            # Fill V
            # V index for B: j*p + k. j in 0..1, k in 0..5.
            # For block b, k in {2*b, 2*b+1}.
            # V indices:
            # j=0, k=2*b -> 0*6 + 2*b = 2*b
            # j=0, k=2*b+1 -> 0*6 + 2*b+1 = 2*b+1
            # j=1, k=2*b -> 1*6 + 2*b = 6+2*b
            # j=1, k=2*b+1 -> 1*6 + 2*b+1 = 6+2*b+1
            v_indices = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_indices[idx]][col] = sc['V'][idx]
                    
            # Fill W
            # W index for C: k*n + i. k in 0..5, i in 0..1.
            # For block b, k in {2*b, 2*b+1}.
            # W indices:
            # k=2*b, i=0 -> 2*b*2 + 0 = 4*b
            # k=2*b, i=1 -> 2*b*2 + 1 = 4*b+1
            # k=2*b+1, i=0 -> (2*b+1)*2 + 0 = 4*b+2
            # k=2*b+1, i=1 -> (2*b+1)*2 + 1 = 4*b+3
            w_indices = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_indices[idx]][col] = sc['W'][idx]

    return {"U": U, "V": V, "W": W}
