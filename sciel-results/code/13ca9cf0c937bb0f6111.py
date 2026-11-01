"""Matrix multiplication 2x2x12 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2x12 matrix multiplication.
    
    Strategy: Use Strassen's algorithm for each 2x2x2 block.
    Split the 12 columns into 6 groups of 2.
    Each 2x2x2 block uses rank 7 (Strassen).
    Total rank = 6 * 7 = 42.
    
    The tensor is M_{i,j,k} for i in [0,1], j in [0,1], k in [0,11].
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (24, rank)
    W has shape (p*n, rank) = (24, rank)
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    rank = 42
    U = np.zeros((n * m, rank))  # 4 x 42
    V = np.zeros((m * p, rank))  # 24 x 42
    W = np.zeros((p * n, rank))  # 24 x 42
    
    # Strassen coefficients for 2x2x2:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # For each Strassen term, we need:
    # - U coefficients (for A matrix elements)
    # - V coefficients (for B matrix elements)
    # - W coefficients (for C matrix elements)
    
    # A is 2x2, indexed as A[i,j] with i in {0,1}, j in {0,1}
    # B is 2x2, indexed as B[j,k] with j in {0,1}, k in {0,1} (local)
    # C is 2x2, indexed as C[i,k] with i in {0,1}, k in {0,1} (local)
    
    # U index: i*m + j = i*2 + j
    # V index: j*p + k (global) = j*12 + (group_start + k_local)
    # W index: k*n + i (global) = (group_start + k_local)*2 + i
    
    # Strassen terms (alpha, beta, gamma):
    # alpha: coefficients for A elements
    # beta: coefficients for B elements
    # gamma: coefficients for C elements
    
    strassen_terms = [
        # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2: (A[1,0] + A[1,1]) * B[0,0]
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3: A[0,0] * (B[0,1] - B[1,1])
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4: A[1,1] * (B[1,0] - B[0,0])
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5: (A[0,0] + A[0,1]) * B[1,1]
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Now construct U, V, W for the full 2x2x12 problem.
    # We have 6 groups of 2 columns each.
    # Group g (g=0,...,5) handles k in {2g, 2g+1}
    
    for g in range(6):
        k_start = 2 * g
        for t in range(7):
            col = g * 7 + t  # column index in U, V, W
            alpha, beta, gamma = strassen_terms[t]
            
            # Fill U: U[i*m + j, col] = alpha[(i,j)]
            for (i, j), coeff in alpha.items():
                U[i * m + j, col] = coeff
            
            # Fill V: V[j*p + k, col] = beta[(j, k_local)]
            for (j, k_local), coeff in beta.items():
                k_global = k_start + k_local
                V[j * p + k_global, col] = coeff
            
            # Fill W: W[k*n + i, col] = gamma[(i, k_local)]
            for (i, k_local), coeff in gamma.items():
                k_global = k_start + k_local
                W[k_global * n + i, col] = coeff
    
    return {"U": U, "V": V, "W": W}
