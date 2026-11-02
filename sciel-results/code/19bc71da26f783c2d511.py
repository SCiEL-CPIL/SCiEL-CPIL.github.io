"""Program for matrix multiplication tensor decomposition (2x2 by 2x10)."""
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition
    for multiplying a (2x2) matrix by a (2x10) matrix.
    
    Uses Strassen's algorithm for 2x2 matrix multiplication (rank 7)
    applied to 5 blocks of 2 columns.
    Total rank = 5 * 7 = 35.
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = 7 * num_blocks
    
    # Strassen decomposition factors for 2x2 case
    # U vectors: coeffs for A[00], A[01], A[10], A[11]
    U_2x2 = np.array([
        [1, 0, 0, 1], 
        [0, 0, 1, 1], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 1, 0, -1] 
    ])
    
    # V vectors: coeffs for B[00], B[01], B[10], B[11]
    V_2x2 = np.array([
        [1, 0, 0, 1], 
        [1, 0, 0, 0], 
        [0, 1, 0, -1],
        [-1, 0, 1, 0], 
        [0, 0, 0, 1], 
        [1, 1, 0, 0], 
        [0, 0, 1, 1]  
    ])
    
    # W vectors: coeffs for C[00], C[10], C[01], C[11]
    # W index in 2x2 case is k*2+i.
    W_2x2 = np.array([
        [1, 0, 0, 1], 
        [0, 1, 0, -1],
        [0, 0, 1, 1], 
        [1, 1, 0, 0], 
        [-1, 0, 1, 0],
        [0, 0, 0, 1], 
        [1, 0, 0, 0]  
    ])
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for t in range(num_blocks):
        for c in range(7):
            # Fill U
            U[:, col] = U_2x2[c]
            
            # Fill V
            # Block t corresponds to columns 2t and 2t+1 of B
            # V index: j*p + k
            # j=0, k=2t -> 2t
            # j=0, k=2t+1 -> 2t+1
            # j=1, k=2t -> 10+2t
            # j=1, k=2t+1 -> 11+2t
            
            v_vals = V_2x2[c]
            V[2*t, col] = v_vals[0]
            V[2*t + 1, col] = v_vals[1]
            V[10 + 2*t, col] = v_vals[2]
            V[11 + 2*t, col] = v_vals[3]
            
            # Fill W
            # W index: k*n + i
            # Block t corresponds to output columns 2t and 2t+1
            # k=2t, i=0 -> 4t
            # k=2t, i=1 -> 4t+1
            # k=2t+1, i=0 -> 4t+2
            # k=2t+1, i=1 -> 4t+3
            
            w_vals = W_2x2[c]
            W[4*t, col] = w_vals[0]
            W[4*t + 1, col] = w_vals[1]
            W[4*t + 2, col] = w_vals[2]
            W[4*t + 3, col] = w_vals[3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
