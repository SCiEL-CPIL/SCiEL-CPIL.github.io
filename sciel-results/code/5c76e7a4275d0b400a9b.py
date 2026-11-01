import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U, V, W vectors for the 7 rank-1 terms
    # Indices: U (0..3), V (0..3), W (0..3)
    # Note: In the full problem, V and W are expanded.
    
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # P7
    ]
    
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(12)]
    W = [[0] * rank for _ in range(12)]
    
    # We have 3 blocks of 2x2 multiplication (for the 6 columns of B/C)
    # Block b corresponds to columns 2b, 2b+1 of B and C
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            # Fill U (same for all blocks, just different columns)
            for idx in range(4):
                if sc['U'][idx] != 0:
                    U[idx][col] = sc['U'][idx]
            
            # Fill V
            # V index for block b, local index idx (0..3)
            # Global V index: j * p + k? No, V is (m x p). m=2, p=6.
            # In the 2x2 sub-problem, V has 4 entries.
            # Local V indices 0,1 correspond to j=0, k=0,1?
            # Let's check the previous code's mapping:
            # v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            # This maps local 0->2b, 1->2b+1, 2->6+2b, 3->6+2b+1.
            # This implies:
            # Local 0: j=0, k=2b
            # Local 1: j=0, k=2b+1
            # Local 2: j=1, k=2b
            # Local 3: j=1, k=2b+1
            # This matches V[j*6 + k].
            
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            for idx in range(4):
                if sc['V'][idx] != 0:
                    V[v_idx[idx]][col] = sc['V'][idx]
            
            # Fill W
            # W index: k*n + i. n=2.
            # Local W indices 0..3.
            # Previous code: w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            # This maps:
            # Local 0 -> 4b
            # Local 1 -> 4b+1
            # Local 2 -> 4b+2
            # Local 3 -> 4b+3
            # Let's check W[k*2 + i].
            # For block b, k can be 2b or 2b+1.
            # If k=2b, indices are 2*(2b)+0 = 4b, 2*(2b)+1 = 4b+1.
            # If k=2b+1, indices are 2*(2b+1)+0 = 4b+2, 2*(2b+1)+1 = 4b+3.
            # So Local 0,1 correspond to k=2b, i=0,1.
            # Local 2,3 correspond to k=2b+1, i=0,1.
            # This matches the mapping.
            
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            for idx in range(4):
                if sc['W'][idx] != 0:
                    W[w_idx[idx]][col] = sc['W'][idx]
    
    return {"U": U, "V": V, "W": W}
