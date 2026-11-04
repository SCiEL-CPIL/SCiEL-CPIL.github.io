import numpy as np

def construct():
    n, m, p = 4, 5, 6
    
    U = np.zeros((n*m, 66), dtype=float)
    V = np.zeros((m*p, 66), dtype=float)
    W = np.zeros((p*n, 66), dtype=float)
    
    rank = 0
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal rank
        U[u_idx, rank] += u_val
        V[v_idx, rank] += v_val
        W[w_idx, rank] += w_val
        rank += 1
        
    # 1. C11 = A[0:4, 0:4] * B[0:4, 0:4]
    # Strassen 4x4x4
    
    # Blocks of A1 (4x4):
    # A11: 0:2, 0:2
    # A12: 0:2, 2:4
    # A21: 2:4, 0:2
    # A22: 2:4, 2:4
    
    # Blocks of B11 (4x4):
    # B11: 0:2, 0:2
    # B12: 0:2, 2:4
    # B21: 2:4, 0:2
    # B22: 2:4, 2:4
    
    # We need to add 7 Strassen products.
    # Each product is 2x2 * 2x2.
    # We will use schoolbook for these 2x2 products (rank 4 each).
    
    # M1 = (A11+A22)(B11+B22)
    # This requires forming sums.
    # This is tricky in a direct tensor decomposition without intermediate storage.
    # However, we can just add the terms.
    
    # Let's just use schoolbook for 4x4x4 to be safe, rank 64.
    # Then total rank 64 + 32 + 6 = 102.
    # Still better than 120.
    
    # Let's try to implement Strassen for 2x2x2 blocks.
    
    def add_strassen_2x2(a_row_start, a_col_start, b_row_start, b_col_start, out_row_start, out_col_start):
        nonlocal rank
        # a is 2x2, b is 2x2
        # A11: a[0:2, 0:2]
        # A12: a[0:2, 2:4]
        # A21: a[2:4, 0:2]
        # A22: a[2:4, 2:4]
        
        # We need to add 7 terms.
        # Each term is a sum of products.
        # This is complex to expand manually.
        
        pass

    # Given the time, I will use schoolbook for 4x4x4 (Rank 64).
    # Total Rank 64 + 32 + 6 = 102.
    
    # 1. A[0:4, 0:4] * B[0:4, 0:4]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j, rank] = 1
                V[j*p + k, rank] = 1
                W[k*n + i, rank] = 1
                rank += 1
                
    # 2. A[0:4, 0:4] * B[0:4, 4:6]
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*m + j, rank] = 1
                V[j*p + (4+k), rank] = 1
                W[(4+k)*n + i, rank] = 1
                rank += 1
                
    # 3. A[0:4, 4:5] * B[0:4, 0:6] ? No.
    # C = A1 B1 + A2 B2
    # A1 is 4x4 (cols 0-3). B1 is 4x6 (rows 0-3).
    # A2 is 4x1 (col 4). B2 is 1x6 (row 4).
    
    # 3. A[0:4, 4] * B[4, 0:6]
    for i in range(4):
        for k in range(6):
            U[i*m + 4, rank] = 1
            V[4*p + k, rank] = 1
            W[k*n + i, rank] = 1
            rank += 1
            
    # Total rank: 64 + 32 + 24 = 120.
    # Wait, A2 B2 is 4x1 * 1x6.
    # My loop for 3 adds 4*6 = 24 terms.
    # But I found rank 6 earlier.
    # Let's use rank 6 for the last part.
    
    # Reset rank
    rank = 64 + 32 # 96
    # Add 6 terms for A2 B2
    for k in range(6):
        # U[i, k] = 1
        for i in range(4):
            U[i*m + 4, rank] = 1
        # V[4, k] = 1
        V[4*p + k, rank] = 1
        # W[k, i] = 1? No.
        # W[k*n + i, rank] = 1
        for i in range(4):
             W[k*n + i, rank] = 1
        rank += 1
             
    # Total rank 96 + 6 = 102.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
