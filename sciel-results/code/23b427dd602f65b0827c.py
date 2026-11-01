"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    This implementation uses a programmatic construction of the Strassen-based
    decomposition. It pre-computes the linear combinations for the A matrix
    and applies them to each 2-column block of the B matrix.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Initialize matrices as lists of lists for the required output format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A indices: 00, 01, 10, 11
    A_coeffs = [
        [1, 0, 0, 1], # A1
        [0, 0, 1, 1], # A2
        [1, 0, 0, 0], # A3
        [0, 0, 0, 1], # A4
        [1, 1, 0, 0], # A5
        [-1, 0, 1, 0],# A6
        [0, 1, 0, -1] # A7
    ]
    
    # Strassen coefficients for B (2x2 block)
    # B indices within block: 00, 01, 10, 11
    B_coeffs = [
        [1, 0, 0, 1], # B1
        [1, 0, 0, 0], # B2
        [0, 1, 0, -1],# B3
        [-1, 0, 1, 0],# B4
        [0, 0, 0, 1], # B5
        [1, 1, 0, 0], # B6
        [0, 0, 1, 1]  # B7
    ]
    
    # Strassen coefficients for C (2x2 block)
    # C indices within block: 00, 10, 01, 11 (matching W convention k*n+i)
    C_coeffs = [
        [1, 0, 0, 1], # C1
        [0, 1, 0, -1],# C2
        [0, 0, 1, 1], # C3
        [1, 1, 0, 0], # C4
        [-1, 0, 1, 0],# C5
        [0, 0, 0, 1], # C6
        [1, 0, 0, 0]  # C7
    ]
    
    col = 0
    for t in range(num_blocks):
        for s in range(7):
            # Fill U (A part)
            for i in range(4):
                U[i][col] = A_coeffs[s][i]
                
            # Fill V (B part)
            # B block t is at columns 2t, 2t+1
            # V indices: j*p + k
            # j=0, k=2t -> 2t
            # j=0, k=2t+1 -> 2t+1
            # j=1, k=2t -> 10+2t
            # j=1, k=2t+1 -> 10+2t+1
            v_vals = B_coeffs[s]
            V[2*t][col] = v_vals[0]
            V[2*t + 1][col] = v_vals[1]
            V[10 + 2*t][col] = v_vals[2]
            V[11 + 2*t][col] = v_vals[3]
            
            # Fill W (C part)
            # C block t is at columns 2t, 2t+1
            # W indices: k*n + i
            # k=2t, i=0 -> 4t
            # k=2t, i=1 -> 4t+1
            # k=2t+1, i=0 -> 4t+2
            # k=2t+1, i=1 -> 4t+3
            w_vals = C_coeffs[s]
            W[4*t][col] = w_vals[0]
            W[4*t + 1][col] = w_vals[1]
            W[4*t + 2][col] = w_vals[2]
            W[4*t + 3][col] = w_vals[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
