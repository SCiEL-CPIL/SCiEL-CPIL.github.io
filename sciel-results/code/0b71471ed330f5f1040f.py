import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4, Rank 16 for 4x4x1
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (c+d)e
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # M3 = a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1
    
    # M4 = d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # M5 = (a+b)h
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (c-a)(e+f)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (b-d)(g+h)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[3][6] = -1
    
    # Define the 7 block products for 4x4x4
    # Each product is defined by:
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign)
    
    # Block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B blocks: 0=B00, 1=B01, 2=B10, 3=B11
    # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
    
    products = [
        # 1. (A00 + A11)(B00 + B11) -> C00 + C11
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)]
        },
        # 2. (A10 + A11)B00 -> C10
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)]
        },
        # 3. A00(B01 - B11) -> C01
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1)]
        },
        # 4. A11(B10 - B00) -> C00 + C10
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)]
        },
        # 5. (A00 + A01)B11 -> C00 - C01
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)]
        },
        # 6. (A10 - A00)(B00 + B01) -> C11
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)]
        },
        # 7. (A01 - A11)(B10 + B11) -> C00 - C11
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1), (3, -1)]
        }
    ]
    
    col = 0
    for prod in products:
        for s in range(7):
            # For each Strassen term s, we add to U, V, W
            for i in range(4):
                if U2[i][s] != 0:
                    for a_idx, a_sign in prod['A']:
                        # Map 2x2 index i to 4x4 index
                        # i is 0,1,2,3 corresponding to a,b,c,d
                        # a=00, b=01, c=10, d=11
                        # Block a_idx is 0,1,2,3 corresponding to A00, A01, A10, A11
                        # The 2x2 matrix entry i corresponds to specific 4x4 entries
                        # We need to map the 2x2 index i to the 4x4 indices within the block
                        # Actually, the 2x2 matrix is formed by the blocks themselves
                        # So U2[i][s] applies to the entire block
                        # Wait, no. The 2x2 matrix has 4 entries.
                        # Each entry is a 2x2 matrix (a block).
                        # So U2[i][s] is the coefficient for the i-th entry of the 2x2 matrix.
                        # The i-th entry of the 2x2 matrix is a 2x2 matrix.
                        # So we need to add U2[i][s] * a_sign to the corresponding 4x4 block.
                        
                        # The 2x2 index i corresponds to:
                        # 0: top-left
                        # 1: top-right
                        # 2: bottom-left
                        # 3: bottom-right
                        # The block a_idx is one of A00, A01, A10, A11
                        # The 4x4 indices for block a_idx are:
                        # If a_idx=0 (A00): rows 0,1 cols 0,1
                        # If a_idx=1 (A01): rows 0,1 cols 2,3
                        # If a_idx=2 (A10): rows 2,3 cols 0,1
                        # If a_idx=3 (A11): rows 2,3 cols 2,3
                        
                        # The 2x2 index i corresponds to:
                        # 0: (0,0) within block
                        # 1: (0,1) within block
                        # 2: (1,0) within block
                        # 3: (1,1) within block
                        
                        # So the 4x4 row is block_row + i_row
                        # The 4x4 col is block_col + i_col
                        
                        block_row = a_idx // 2
                        block_col = a_idx % 2
                        i_row = i // 2
                        i_col = i % 2
                        r = block_row * 2 + i_row
                        c = block_col * 2 + i_col
                        U[r * m + c][col] += U2[i][s] * a_sign
            for j in range(4):
                if V2[j][s] != 0:
                    for b_idx, b_sign in prod['B']:
                        block_row = b_idx // 2
                        block_col = b_idx % 2
                        j_row = j // 2
                        j_col = j % 2
                        r = block_row * 2 + j_row
                        c = block_col * 2 + j_col
                        V[r * p + c][col] += V2[j][s] * b_sign
            for k in range(4):
                if W2[k][s] != 0:
                    for c_idx, c_sign in prod['C']:
                        block_row = c_idx // 2
                        block_col = c_idx % 2
                        k_row = k // 2
                        k_col = k % 2
                        r = block_row * 2 + k_row
                        c = block_col * 2 + k_col
                        # W index is k*n + i, where k is col of B, i is row of A
                        # Here c_idx is the block index in C
                        # C is 4x4, so c_idx 0,1,2,3
                        # c_idx 0: C00 (rows 0,1 cols 0,1)
                        # c_idx 1: C01 (rows 0,1 cols 2,3)
                        # c_idx 2: C10 (rows 2,3 cols 0,1)
                        # c_idx 3: C11 (rows 2,3 cols 2,3)
                        # W index: col * n + row
                        # col = block_col * 2 + k_col
                        # row = block_row * 2 + k_row
                        w_row = block_row * 2 + k_row
                        w_col = block_col * 2 + k_col
                        W[w_col * n + w_row][col] += W2[k][s] * c_sign
            col += 1
            
    # Add rank 16 for 4x4x1 (last column k=4)
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
