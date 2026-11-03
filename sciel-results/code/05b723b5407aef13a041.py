import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Helper to create 2x2x2 Strassen decomposition
    def strassen_2x2x2():
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1: (A00+A11)(B00+B11)
        # U: A00, A11
        U[0, 0] = 1; U[3, 0] = 1
        # V: B00, B11
        V[0, 0] = 1; V[3, 0] = 1
        # W: C00, C11
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A10+A11)B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1 # C10
        
        # M3: A00(B10-B11)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1 # C01
        
        # M4: A11(B01-B00)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[3, 3] = 1 # C11
        
        # M5: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1 # C01
        
        # M6: (A10-A00)(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[2, 5] = 1 # C10
        
        # M7: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1 # C00
        
        return U, V, W

    U2, V2, W2 = strassen_2x2x2()
    
    # Define the 7 outer products for 4x4x4
    # Each product q has:
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign) -- where block_idx is 0..3 for C
    # C00, C01, C10, C11
    
    # Block indices:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    
    outer_products = [
        # M1: (A00+A11)(B00+B11) -> C00+C11
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # M2: (A10+A11)B00 -> C10
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)]
        },
        # M3: A00(B10-B11) -> C01
        {
            'A': [(0, 1)],
            'B': [(2, 1), (3, -1)],
            'C': [(1, 1)]
        },
        # M4: A11(B01-B00) -> C11
        {
            'A': [(3, 1)],
            'B': [(1, 1), (0, -1)],
            'C': [(3, 1)]
        },
        # M5: (A00+A01)B11 -> C01
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(1, 1)]
        },
        # M6: (A10-A00)(B00+B01) -> C10
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(2, 1)]
        },
        # M7: (A01-A11)(B10+B11) -> C00
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1)]
        }
    ]
    
    # Build 4x4x4 decomposition
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for q, prod in enumerate(outer_products):
        col_offset = q * 7
        for t in range(7):
            # Add U pattern
            for block_idx, sign in prod['A']:
                bi, bj = divmod(block_idx, 2)
                # Rows in U4: i*4 + j
                # Local rows in U2: i*2 + j
                for li in range(2):
                    for lj in range(2):
                        global_i = bi * 2 + li
                        global_j = bj * 2 + lj
                        u_row = global_i * 4 + global_j
                        u_val = U2[li * 2 + lj, t]
                        if u_val != 0:
                            U4[u_row, col_offset + t] += sign * u_val
                            
            # Add V pattern
            for block_idx, sign in prod['B']:
                bi, bj = divmod(block_idx, 2)
                # Rows in V4: j*4 + k (local)
                # Local rows in V2: j*2 + k
                for lj in range(2):
                    for lk in range(2):
                        global_j = bi * 2 + lj
                        global_k = bj * 2 + lk
                        v_row = global_j * 4 + global_k
                        v_val = V2[lj * 2 + lk, t]
                        if v_val != 0:
                            V4[v_row, col_offset + t] += sign * v_val
                            
            # Add W pattern
            for block_idx, sign in prod['C']:
                bi, bj = divmod(block_idx, 2)
                # Rows in W4: k*4 + i (local)
                # Local rows in W2: k*2 + i
                for lk in range(2):
                    for li in range(2):
                        global_k = bi * 2 + lk
                        global_i = bj * 2 + li
                        w_row = global_k * 4 + global_i
                        w_val = W2[lk * 2 + li, t]
                        if w_val != 0:
                            W4[w_row, col_offset + t] += sign * w_val

    # Now combine for 4x4x9
    # Part 1: Cols 0-3 (Strassen 4x4x4)
    # Part 2: Cols 4-7 (Strassen 4x4x4)
    # Part 3: Col 8 (Schoolbook 4x4x1)
    
    R = 49 + 49 + 16
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((36, R), dtype=int)
    W = np.zeros((36, R), dtype=int)
    
    # Part 1
    U[:, 0:49] = U4
    # V4 has shape 16x49 (rows j*4+k, k in 0..3)
    # V has shape 36xR (rows j*9+k)
    # We need to map V4 rows j*4+k to V rows j*9+k
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = V4[j*4 + k, 0:49]
    # W4 has shape 16x49 (rows k*4+i, k in 0..3)
    # W has shape 36xR (rows k*9+i? NO, W index is k*n+i. n=4. So k*4+i)
    # Wait, the problem statement says W[k*n+i, r].
    # In the 4x4x4 part, k is 0..3. So rows are 0..15.
    # In the 4x4x9 part, k is 0..8. So rows are 0..35.
    # So W4 rows map directly to W rows 0..15.
    W[0:16, 0:49] = W4
    
    # Part 2
    U[:, 49:98] = U4
    # V4 rows j*4+k (k in 0..3) map to V rows j*9 + (k+4)
    for j in range(4):
        for k in range(4):
            V[j*9 + (k+4), 49:98] = V4[j*4 + k, 0:49]
    # W4 rows k*4+i (k in 0..3) map to W rows (k+4)*4 + i
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, 0:49]
            
    # Part 3: Schoolbook for col 8
    # 16 terms
    for i in range(4):
        for j in range(4):
            col = 98 + i*4 + j
            U[i*4 + j, col] = 1
            V[j*9 + 8, col] = 1
            W[8*4 + i, col] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
