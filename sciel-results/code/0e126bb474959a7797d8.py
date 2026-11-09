import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin

    n, m, p = DIMENSIONS

    # Strategy: Kronecker product decomposition
    # Factorize (3, 4, 8) into (3, 2, 4) and (1, 2, 2)
    # T_{3,4,8} = T_{3,2,4} \otimes T_{1,2,2}
    # Rank = Rank(T_{3,2,4}) * Rank(T_{1,2,2})
    # Using naive for both: 24 * 4 = 96
    
    # Sub-problem 1: 3 x 2 x 4
    n1, m1, p1 = 3, 2, 4
    rank1 = n1 * m1 * p1  # 24
    
    # Sub-problem 2: 1 x 2 x 2
    n2, m2, p2 = 1, 2, 2
    rank2 = n2 * m2 * p2  # 4
    
    total_rank = rank1 * rank2  # 96
    
    # Initialize U, V, W for the full problem
    # U: (n*m) x rank  -> (3*4) x 96 = 12 x 96
    # V: (m*p) x rank  -> (4*8) x 96 = 32 x 96
    # W: (p*n) x rank  -> (8*3) x 96 = 24 x 96
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Construct U1, V1, W1 for sub-problem 1 (3x2x4) - naive
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
                
    # Construct U2, V2, W2 for sub-problem 2 (1x2x2) - naive
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
    # For each rank index c = c1 * rank2 + c2:
    # The non-zero indices in U1, V1, W1 for column c1 are unique (naive)
    # Similarly for U2, V2, W2 for column c2
    
    # In naive decomposition:
    # c1 = i1 * (m1 * p1) + j1 * p1 + k1
    # c2 = i2 * (m2 * p2) + j2 * p2 + k2
    
    # Full indices:
    # U index: i_full = i1 * n2 + i2, j_full = j1 * m2 + j2
    #          u_idx = i_full * m + j_full = (i1 * n2 + i2) * m + (j1 * m2 + j2)
    #          But m = m1 * m2, so:
    #          u_idx = (i1 * n2 + i2) * (m1 * m2) + (j1 * m2 + j2)
    #                  = i1 * n2 * m1 * m2 + i2 * m1 * m2 + j1 * m2 + j2
    # V index: j_full = j1 * m2 + j2, k_full = k1 * p2 + k2
    #          v_idx = j_full * p + k_full = (j1 * m2 + j2) * (p1 * p2) + (k1 * p2 + k2)
    # W index: k_full = k1 * p2 + k2, i_full = i1 * n2 + i2
    #          w_idx = k_full * n + i_full = (k1 * p2 + k2) * (n1 * n2) + (i1 * n2 + i2)
    
    for c1 in range(rank1):
        # Decode c1 into (i1, j1, k1)
        i1 = c1 // (m1 * p1)
        rem1 = c1 % (m1 * p1)
        j1 = rem1 // p1
        k1 = rem1 % p1
        
        for c2 in range(rank2):
            # Decode c2 into (i2, j2, k2)
            i2 = c2 // (m2 * p2)
            rem2 = c2 % (m2 * p2)
            j2 = rem2 // p2
            k2 = rem2 % p2
            
            col_full = c1 * rank2 + c2
            
            # Calculate full indices
            u_idx = (i1 * n2 + i2) * m + (j1 * m2 + j2)
            v_idx = (j1 * m2 + j2) * p + (k1 * p2 + k2)
            w_idx = (k1 * p2 + k2) * n + (i1 * n2 + i2)
            
            U[u_idx][col_full] = 1
            V[v_idx][col_full] = 1
            W[w_idx][col_full] = 1
            
    # Check deadline
    if time.time() > deadline:
        pass
        
    return {"U": U, "V": V, "W": W}
