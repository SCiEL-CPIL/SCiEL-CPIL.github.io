"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 21 using Strassen's algorithm for 3 independent 2x2x2 blocks
    rank = 21
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Terms M1..M7
    # U contributions (A matrix)
    # V contributions (B matrix)
    # W contributions (C matrix)
    
    # Local indices for A (i,j): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Local indices for B (j,k): 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block)
    # Local indices for C (i,k): 00, 01, 10, 11 -> 0, 1, 2, 3 (relative to block)
    
    # Define the 7 Strassen terms
    # Each term is a dict of {matrix: {local_idx: coeff}}
    # We need to map local idx to global idx for V and W based on block t
    
    strassen_terms = [
        # M1: (a00+a11)(b00+b11)
        {
            'U': {0: 1, 3: 1}, # a00, a11
            'V': {0: 1, 3: 1}, # b00, b11
            'W': {0: 1, 3: 1}  # c00, c11
        },
        # M2: (a10+a11)b00
        {
            'U': {2: 1, 3: 1}, # a10, a11
            'V': {0: 1},       # b00
            'W': {1: 1, 3: -1} # c10, c11
        },
        # M3: a00(b01-b11)
        {
            'U': {0: 1},       # a00
            'V': {1: 1, 3: -1},# b01, b11
            'W': {2: 1, 3: 1}  # c01, c11
        },
        # M4: a11(b10-b00)
        {
            'U': {3: 1},       # a11
            'V': {2: 1, 0: -1},# b10, b00
            'W': {0: 1, 1: 1}  # c00, c10
        },
        # M5: (a00+a01)b11
        {
            'U': {0: 1, 1: 1}, # a00, a01
            'V': {3: 1},       # b11
            'W': {0: -1, 2: 1} # c00, c01
        },
        # M6: (a10-a00)(b00+b01)
        {
            'U': {2: 1, 0: -1},# a10, a00
            'V': {0: 1, 1: 1}, # b00, b01
            'W': {3: 1}        # c11
        },
        # M7: (a01-a11)(b10+b11)
        {
            'U': {1: 1, 3: -1},# a01, a11
            'V': {2: 1, 3: 1}, # b10, b11
            'W': {0: 1}        # c00
        }
    ]
    
    # Fill U, V, W
    for t in range(3): # 3 blocks of 2x2
        base_col = t * 7
        for s in range(7):
            col = base_col + s
            term = strassen_terms[s]
            
            # U: indices are global i*m+j, same as local for 2x2
            for local_idx, coeff in term['U'].items():
                U[local_idx, col] = coeff
            
            # V: indices are global j*p+k
            # Local b_jk: j in {0,1}, k in {0,1}
            # Global k = 2*t + local_k
            for local_idx, coeff in term['V'].items():
                local_j = local_idx // 2
                local_k = local_idx % 2
                global_k = 2 * t + local_k
                global_idx = local_j * p + global_k
                V[global_idx, col] = coeff
            
            # W: indices are global k*n+i
            # Local c_ik: i in {0,1}, k in {0,1}
            # Global k = 2*t + local_k
            for local_idx, coeff in term['W'].items():
                local_i = local_idx // 2
                local_k = local_idx % 2
                global_k = 2 * t + local_k
                global_idx = global_k * n + local_i
                W[global_idx, col] = coeff

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
