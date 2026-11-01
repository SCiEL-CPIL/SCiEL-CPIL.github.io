import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 54
    R = 54
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # --- Block 1: 2x1x6 (Rank 12) ---
    # j=0
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # col is now 12

    # --- Block 2: 2x4x6 (Rank 42) ---
    # Split into two 2x2x6 blocks: j in {1,2} and j in {3,4}
    
    # Strassen coefficients for 2x2x2
    # u, v, w are lists of 7 vectors of length 4
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_u = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_v = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [-1, 0, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_w = [
        [1, 0, 0, 1], # M1 (w00=1, w11=1)
        [0, 1, 0, -1],# M2 (w01=1, w11=-1)
        [0, 0, 1, 1], # M3 (w10=1, w11=1)
        [1, 1, 0, 0], # M4 (w00=1, w01=1)
        [-1, 0, 1, 0],# M5 (w00=-1, w10=1)
        [0, 0, 0, 1], # M6 (w11=1)
        [1, 0, 0, 0]  # M7 (w00=1)
    ]
    
    def add_strassen_block(j_offset, k_offset, start_col):
        # j_offset: 1 or 3
        # k_offset: 0, 2, or 4
        # start_col: starting column index
        for s in range(7):
            c = start_col + s
            # Local indices i' in {0,1}, j' in {0,1}, k' in {0,1}
            for i_prime in range(2):
                for j_prime in range(2):
                    val = strassen_u[s][j_prime * 2 + i_prime] # Wait, u index is i,j.
                    # My strassen_u list is indexed by [i][j]?
                    # No, I flattened it.
                    # Let's check flattening order.
                    # I used [1, 0, 0, 1] for M1.
                    # A00=1, A01=0, A10=0, A11=1.
                    # So index 0: A00, 1: A01, 2: A10, 3: A11.
                    # Index = i * 2 + j.
                    val_u = strassen_u[s][i_prime * 2 + j_prime]
                    if val_u != 0:
                        global_i = i_prime
                        global_j = j_offset + j_prime
                        U[global_i * m + global_j][c] = val_u
                        
            for j_prime in range(2):
                for k_prime in range(2):
                    val_v = strassen_v[s][j_prime * 2 + k_prime]
                    if val_v != 0:
                        global_j = j_offset + j_prime
                        global_k = k_offset + k_prime
                        V[global_j * p + global_k][c] = val_v
                        
            for k_prime in range(2):
                for i_prime in range(2):
                    val_w = strassen_w[s][k_prime * 2 + i_prime]
                    if val_w != 0:
                        global_k = k_offset + k_prime
                        global_i = i_prime
                        W[global_k * n + global_i][c] = val_w
                        
        return start_col + 7

    # Block 2a: j in {1, 2}
    # B parts: k in {0,1}, {2,3}, {4,5}
    c = 12
    c = add_strassen_block(1, 0, c)
    c = add_strassen_block(1, 2, c)
    c = add_strassen_block(1, 4, c)
    
    # Block 2b: j in {3, 4}
    # B parts: k in {0,1}, {2,3}, {4,5}
    c = add_strassen_block(3, 0, c)
    c = add_strassen_block(3, 2, c)
    c = add_strassen_block(3, 4, c)
    
    return {"U": U, "V": V, "W": W}
