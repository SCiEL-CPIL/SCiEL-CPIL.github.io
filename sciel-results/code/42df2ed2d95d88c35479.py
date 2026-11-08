import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Strategy: Kronecker product decomposition
    # We factorize the dimensions (3, 4, 8) into (3, 2, 4) and (1, 2, 2)
    # T_{3,4,8} = T_{3,2,4} \otimes T_{1,2,2}
    # Rank(T_{3,4,8}) = Rank(T_{3,2,4}) * Rank(T_{1,2,2})
    
    # Sub-problem 1: 3 x 2 x 4
    n1, m1, p1 = 3, 2, 4
    rank1 = n1 * m1 * p1
    
    # Sub-problem 2: 1 x 2 x 2
    n2, m2, p2 = 1, 2, 2
    rank2 = n2 * m2 * p2
    
    total_rank = rank1 * rank2
    
    # Initialize U, V, W for the full problem
    # U: (n*m) x rank  -> (3*4) x 96
    # V: (m*p) x rank  -> (4*8) x 96
    # W: (p*n) x rank  -> (8*3) x 96
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Construct U1, V1, W1 for sub-problem 1 (3x2x4)
    U1 = [[0] * rank1 for _ in range(n1 * m1)]
    V1 = [[0] * rank1 for _ in range(m1 * p1)]
    W1 = [[0] * rank1 for _ in range(p1 * n1)]
    
    col1 = 0
    for i in range(n1):
        for j in range(m1):
            for k in range(p1):
                U1[i * m1 + j][col1] = 1
                V1[j * p1 + k][col1] = 1
                W1[k * n1 + i][col1] = 1
                col1 += 1
                
    # Construct U2, V2, W2 for sub-problem 2 (1x2x2)
    U2 = [[0] * rank2 for _ in range(n2 * m2)]
    V2 = [[0] * rank2 for _ in range(m2 * p2)]
    W2 = [[0] * rank2 for _ in range(p2 * n2)]
    
    col2 = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U2[i * m2 + j][col2] = 1
                V2[j * p2 + k][col2] = 1
                W2[k * n2 + i][col2] = 1
                col2 += 1
                
    # Combine via Kronecker product logic
    # The index for the full tensor is (i1, i2) for U, etc.
    # U index: i_full = i1 * n2 + i2  (since n = n1*n2)
    # V index: j_full = j1 * m2 + j2  (since m = m1*m2)
    # W index: k_full = k1 * p2 + k2  (since p = p1*p2)
    
    # Iterate through all combinations of rank1 and rank2 indices
    for c1 in range(rank1):
        for c2 in range(rank2):
            col_full = c1 * rank2 + c2
            
            # Find non-zero entries in U1, V1, W1 for column c1
            # And U2, V2, W2 for column c2
            # Since they are sparse (one 1 per column in naive), we can just find the indices
            
            # U1 indices
            i1_idx = -1
            for r in range(n1 * m1):
                if U1[r][c1] != 0:
                    i1_idx = r
                    break
            # U2 indices
            i2_idx = -1
            for r in range(n2 * m2):
                if U2[r][c2] != 0:
                    i2_idx = r
                    break
            
            # V1 indices
            j1_idx = -1
            for r in range(m1 * p1):
                if V1[r][c1] != 0:
                    j1_idx = r
                    break
            # V2 indices
            j2_idx = -1
            for r in range(m2 * p2):
                if V2[r][c2] != 0:
                    j2_idx = r
                    break
            
            # W1 indices
            k1_idx = -1
            for r in range(p1 * n1):
                if W1[r][c1] != 0:
                    k1_idx = r
                    break
            # W2 indices
            k2_idx = -1
            for r in range(p2 * n2):
                if W2[r][c2] != 0:
                    k2_idx = r
                    break
            
            if i1_idx != -1 and i2_idx != -1 and j1_idx != -1 and j2_idx != -1 and k1_idx != -1 and k2_idx != -1:
                # Calculate full indices
                # n = n1 * n2, m = m1 * m2, p = p1 * p2
                # U index: i1 * n2 + i2
                u_idx = i1_idx * n2 + i2_idx
                # V index: j1 * m2 + j2
                v_idx = j1_idx * m2 + j2_idx
                # W index: k1 * p2 + k2
                w_idx = k1_idx * p2 + k2_idx
                
                U[u_idx][col_full] = U1[i1_idx][c1] * U2[i2_idx][c2]
                V[v_idx][col_full] = V1[j1_idx][c1] * V2[j2_idx][c2]
                W[w_idx][col_full] = W1[k1_idx][c1] * W2[k2_idx][c2]
                
    # Check deadline
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
