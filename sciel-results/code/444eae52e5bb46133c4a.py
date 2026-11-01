import time
import numpy as np

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 25
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2x2: (u_coeffs, v_coeffs, w_coeffs)
    # u: [a, b, c, d], v: [e, f, g, h], w: [C00, C10, C01, C11]
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),   # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),   # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])    # M7
    ]
    
    col = 0
    
    # Process 3 Strassen blocks for k=0,1; 2,3; 4,5
    for block_idx in range(3):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        for u_loc, v_loc, w_loc in strassen_terms:
            # Map U: a=idx0, b=idx1, c=idx2, d=idx3
            U[0, col] = u_loc[0]
            U[1, col] = u_loc[1]
            U[2, col] = u_loc[2]
            U[3, col] = u_loc[3]
            
            # Map V: e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
            V[0 * p + k0, col] = v_loc[0]
            V[0 * p + k1, col] = v_loc[1]
            V[1 * p + k0, col] = v_loc[2]
            V[1 * p + k1, col] = v_loc[3]
            
            # Map W: C00=W[2*k0+0], C10=W[2*k0+1], C01=W[2*k1+0], C11=W[2*k1+1]
            W[k0 * n + 0, col] = w_loc[0]
            W[k0 * n + 1, col] = w_loc[1]
            W[k1 * n + 0, col] = w_loc[2]
            W[k1 * n + 1, col] = w_loc[3]
            
            col += 1
    
    # Process 1 Schoolbook block for k=6
    k0 = 6
    
    # Term 1: a * B[0,6] -> C[0,6]
    U[0, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    col += 1
    
    # Term 2: b * B[1,6] -> C[0,6]
    U[1, col] = 1
    V[1 * p + 6, col] = 1
    W[6 * n + 0, col] = 1
    col += 1
    
    # Term 3: c * B[0,6] -> C[1,6]
    U[2, col] = 1
    V[0 * p + 6, col] = 1
    W[6 * n + 1, col] = 1
    col += 1
    
    # Term 4: d * B[1,6] -> C[1,6]
    U[3, col] = 1
    V[1 * p + 6, col] = 1
    W[6 * n + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
