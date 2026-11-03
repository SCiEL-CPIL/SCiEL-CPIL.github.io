import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 22
    rank = 22
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 vectors for M_2,3,2
    # u: size 6 (2*3), v: size 6 (3*2), w: size 4 (2*2)
    u_vecs = [
        [1, 0, 0, 0, 1, 0], # 1
        [0, 0, 0, 1, 1, 0], # 2
        [1, 0, 0, 0, 0, 0], # 3
        [0, 0, 0, 0, 1, 0], # 4
        [1, 1, 0, 0, 0, 0], # 5
        [-1, 0, 0, 1, 0, 0], # 6
        [0, 1, 0, -1, 0, 0], # 7 (A_01 - A_11)
        [0, 0, 1, 0, 0, 0], # 8
        [0, 0, 1, 0, 0, 0], # 9
        [0, 0, 0, 0, 0, 1], # 10
        [0, 0, 0, 0, 0, 1]  # 11
    ]
    
    v_vecs = [
        [1, 0, 0, 1, 0, 0], # 1 (B_00, B_11)
        [1, 0, 0, 0, 0, 0], # 2 (B_00)
        [0, 1, 0, -1, 0, 0], # 3 (B_01, -B_11)
        [-1, 0, 0, 1, 0, 0], # 4 (-B_00, B_10)
        [0, 0, 0, 0, 1, 0], # 5 (B_11)
        [1, 1, 0, 0, 0, 0], # 6 (B_00, B_01)
        [0, 0, 0, 1, 1, 0], # 7 (B_10, B_11)
        [0, 0, 0, 0, 1, 0], # 8 (B_20) -> idx 4
        [0, 0, 0, 0, 0, 1], # 9 (B_21) -> idx 5
        [0, 0, 0, 0, 1, 0], # 10 (B_20)
        [0, 0, 0, 0, 0, 1]  # 11 (B_21)
    ]
    
    w_vecs = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, -1], # 2
        [0, 0, 1, 0], # 3
        [1, 0, 0, 1], # 4
        [-1, 0, 0, 1], # 5
        [0, 1, 0, 0], # 6
        [1, 1, 0, 0], # 7
        [1, 0, 0, 0], # 8
        [0, 0, 1, 0], # 9
        [0, 1, 0, 0], # 10
        [0, 0, 0, 1]  # 11
    ]
    
    # Map to M_2,3,4
    # Block 1: k=0,1
    for c in range(11):
        col1 = c
        # U is same
        for i in range(2):
            for j in range(3):
                U[i*m + j][col1] = u_vecs[c][i*m + j]
        
        # V mapping: small idx 2j+k -> large idx 4j+k
        for j in range(3):
            for k in range(2):
                small_idx = j*2 + k
                large_idx = j*4 + k
                V[large_idx][col1] = v_vecs[c][small_idx]
        
        # W mapping: small idx 2k+i -> large idx 2k+i (same)
        for k in range(2):
            for i in range(2):
                small_idx = k*2 + i
                large_idx = k*2 + i
                W[large_idx][col1] = w_vecs[c][small_idx]
                
    # Block 2: k=2,3 (mapped from k'=0,1)
    for c in range(11):
        col2 = 11 + c
        # U is same
        for i in range(2):
            for j in range(3):
                U[i*m + j][col2] = u_vecs[c][i*m + j]
        
        # V mapping: small idx 2j+k' -> large idx 4j+(k'+2)
        for j in range(3):
            for k_prime in range(2):
                small_idx = j*2 + k_prime
                large_idx = j*4 + (k_prime + 2)
                V[large_idx][col2] = v_vecs[c][small_idx]
        
        # W mapping: small idx 2k'+i -> large idx 2(k'+2)+i
        for k_prime in range(2):
            for i in range(2):
                small_idx = k_prime*2 + i
                large_idx = (k_prime + 2)*2 + i
                W[large_idx][col2] = w_vecs[c][small_idx]

    return {"U": U, "V": V, "W": W}
