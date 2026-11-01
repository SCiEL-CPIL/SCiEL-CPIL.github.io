import numpy as np

DIMENSIONS = [2, 2, 15]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build the rank 53 decomposition.
    # 7 blocks of 2x2 (Strassen, rank 7)
    # 1 block of 2x1 (Schoolbook, rank 4)
    
    total_rank = 7 * 7 + 4
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col_offset = 0
    
    # Define Strassen terms for 2x2x2
    # U is 4x1, V is 4x1 (flattened 2x2), W is 4x1 (flattened 2x2)
    # Local indices for U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Local indices for W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Wait, W is k*n+i.
    # Local k=0,1. i=0,1.
    # W[0] is (k=0, i=0) -> C[0,0]
    # W[1] is (k=0, i=1) -> C[1,0]
    # W[2] is (k=1, i=0) -> C[0,1]
    # W[3] is (k=1, i=1) -> C[1,1]
    
    strassen_terms = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [0, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, -1]) # M7
    ]
    
    # Process 7 blocks of 2 columns
    for b in range(7):
        k1 = b * 2
        k2 = b * 2 + 1
        # k1, k2 are the global column indices in B and C
        
        for t in range(7):
            u_val = strassen_terms[t][0]
            v_val = strassen_terms[t][1]
            w_val = strassen_terms[t][2]
            
            # Fill U
            for idx in range(4):
                if u_val[idx] != 0:
                    U[idx, col_offset] = u_val[idx]
            
            # Fill V
            # V is m x p = 2 x 15.
            # Local V is 2x2.
            # Local (j, k') -> Global (j, k_global)
            # k'=0 -> k1, k'=1 -> k2
            for j in range(2):
                for kp in range(2):
                    val = v_val[j * 2 + kp]
                    if val != 0:
                        k_global = k1 if kp == 0 else k2
                        V[j * p + k_global, col_offset] = val
                        
            # Fill W
            # W is p x n = 15 x 2.
            # Local W is 2x2 (k', i).
            # Local (k', i) -> Global (k_global, i)
            for kp in range(2):
                for i in range(2):
                    val = w_val[kp * 2 + i]
                    if val != 0:
                        k_global = k1 if kp == 0 else k2
                        W[k_global * n + i, col_offset] = val
                        
            col_offset += 1
            
    # Process 1 block of 1 column
    k1 = 14
    # Standard 2x2x1 terms
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    single_terms = [
        # (U, V, W)
        # V is 2x1 local. W is 1x2 local.
        ([1, 0, 0, 0], [1, 0], [1, 0]), # A00 B00 -> C00
        ([0, 1, 0, 0], [0, 1], [1, 0]), # A01 B10 -> C00
        ([0, 0, 1, 0], [1, 0], [0, 1]), # A10 B00 -> C10
        ([0, 0, 0, 1], [0, 1], [0, 1])  # A11 B10 -> C10
    ]
    
    for t in range(4):
        u_val = single_terms[t][0]
        v_val = single_terms[t][1]
        w_val = single_terms[t][2]
        
        # Fill U
        for idx in range(4):
            if u_val[idx] != 0:
                U[idx, col_offset] = u_val[idx]
                
        # Fill V
        # V is 2x1 local.
        for j in range(2):
            val = v_val[j]
            if val != 0:
                V[j * p + k1, col_offset] = val
                
        # Fill W
        # W is 1x2 local.
        for i in range(2):
            val = w_val[i]
            if val != 0:
                W[k1 * n + i, col_offset] = val
                
        col_offset += 1

    return {"U": U, "V": V, "W": W}
