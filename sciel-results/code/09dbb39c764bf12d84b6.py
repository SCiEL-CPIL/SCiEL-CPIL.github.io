import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # U shape: (2*4, rank) = (8, rank)
    # V shape: (4*5, rank) = (20, rank)
    # W shape: (5*2, rank) = (10, rank)
    
    # We construct rank 32 solution
    # 4 blocks of 2x2x2 (rank 7 each)
    # 2 blocks of 2x2x1 (rank 2 each)
    
    # Total rank = 4*7 + 2*2 = 32
    
    R = 32
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    rank_offset = 0
    
    # Strassen coefficients for A (2x2), B (2x2), C (2x2)
    strassen_terms = [
        # M1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # M2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # M3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # M4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # M5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # M6
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # M7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1})
    ]
    
    def add_strassen_block(j_off, k_off):
        nonlocal rank_offset
        for term in strassen_terms:
            a_coeffs, b_coeffs, c_coeffs = term
            col = rank_offset
            
            for (i, j), val in a_coeffs.items():
                u_idx = i * m + (j_off + j)
                U[u_idx, col] = val
                
            for (j, k), val in b_coeffs.items():
                v_idx = (j_off + j) * p + (k_off + k)
                V[v_idx, col] = val
                
            for (k, i), val in c_coeffs.items():
                w_idx = (k_off + k) * n + i
                W[w_idx, col] = val
                
            rank_offset += 1

    def add_simple_block(j_off, k_off):
        nonlocal rank_offset
        
        # Term 1: A[:, j_off] * B[j_off, k_off]
        col = rank_offset
        U[0*m + j_off, col] = 1
        U[1*m + j_off, col] = 1
        V[j_off*p + k_off, col] = 1
        W[k_off*n + 0, col] = 1
        W[k_off*n + 1, col] = 1
        rank_offset += 1
        
        # Term 2: A[:, j_off+1] * B[j_off+1, k_off]
        col = rank_offset
        U[0*m + j_off + 1, col] = 1
        U[1*m + j_off + 1, col] = 1
        V[(j_off+1)*p + k_off, col] = 1
        W[k_off*n + 0, col] = 1
        W[k_off*n + 1, col] = 1
        rank_offset += 1

    # Block 1: j in 0,1; k in 0,1
    add_strassen_block(0, 0)
    # Block 2: j in 2,3; k in 0,1
    add_strassen_block(2, 0)
    # Block 3: j in 0,1; k in 2,3
    add_strassen_block(0, 2)
    # Block 4: j in 2,3; k in 2,3
    add_strassen_block(2, 2)
    
    # Block 5: j in 0,1; k in 4
    add_simple_block(0, 4)
    # Block 6: j in 2,3; k in 4
    add_simple_block(2, 4)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
