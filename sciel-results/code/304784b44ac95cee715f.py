"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    
    # Define the Strassen factors for 2x2 matrix multiplication
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2 correspond to a00, a01, a10, a11
    # Rows of V2 correspond to b00, b01, b10, b11
    # Rows of W2 correspond to c00, c01, c10, c11
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # b00
        [0, 0, 1, 0, 0, 1, 0],   # b01
        [0, 0, 0, 1, 0, 0, 1],   # b10
        [1, 0, -1, 0, 1, 0, 1]   # b11
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # c00
        [0, 0, 1, 0, 1, 0, 0],   # c01
        [0, 1, 0, 1, 0, 0, 0],   # c10
        [1, -1, 1, 0, 0, 1, 0]   # c11
    ])
    
    # Construct 4x4 factors using Kronecker-like product with specific indexing
    # Rank is 7 * 7 = 49
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to get index in 2x2 factor matrix
    # For a 4x4 matrix element (i, j), block indices are I=i//2, J=j//2
    # Local indices are i'=i%2, j'=j%2
    # Row in 2x2 factor for block part: 2*I + J
    # Row in 2x2 factor for local part: 2*i' + j'
    
    for i in range(n):
        for j in range(m):
            I = i // 2
            J = j // 2
            ip = i % 2
            jp = j % 2
            
            u_block_idx = 2 * I + J
            u_local_idx = 2 * ip + jp
            
            for r in range(7):
                for s in range(7):
                    col = 7 * r + s
                    U[i * m + j, col] = U2[u_block_idx, r] * U2[u_local_idx, s]
                    
    for j in range(m):
        for k in range(p):
            J = j // 2
            K = k // 2
            jp = j % 2
            kp = k % 2
            
            v_block_idx = 2 * J + K
            v_local_idx = 2 * jp + kp
            
            for r in range(7):
                for s in range(7):
                    col = 7 * r + s
                    V[j * p + k, col] = V2[v_block_idx, r] * V2[v_local_idx, s]
                    
    for k in range(p):
        for i in range(n):
            K = k // 2
            I = i // 2
            kp = k % 2
            ip = i % 2
            
            w_block_idx = 2 * K + I
            w_local_idx = 2 * kp + ip
            
            for r in range(7):
                for s in range(7):
                    col = 7 * r + s
                    W[k * n + i, col] = W2[w_block_idx, r] * W2[w_local_idx, s]
                    
    # Convert to lists of lists as required by the interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
