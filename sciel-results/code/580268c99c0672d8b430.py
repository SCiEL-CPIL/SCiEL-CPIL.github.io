import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    # p is 6, so we have 3 blocks of size 2
    num_blocks = p // 2
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    # A = [[a, b], [c, d]]
    # L1 = a+d, L2 = c+d, L3 = a, L4 = d, L5 = a+b, L6 = c-a, L7 = b-d
    # Indices in U: 0=a, 1=b, 2=c, 3=d (row-major)
    # U[row, col]
    # Row 0: a (0,0), Row 1: b (0,1), Row 2: c (1,0), Row 3: d (1,1)
    
    # Coefficients for the 7 products
    # P1: a+d -> [1, 0, 0, 1]
    # P2: c+d -> [0, 0, 1, 1]
    # P3: a   -> [1, 0, 0, 0]
    # P4: d   -> [0, 0, 0, 1]
    # P5: a+b -> [1, 1, 0, 0]
    # P6: c-a -> [-1, 0, 1, 0]
    # P7: b-d -> [0, 1, 0, -1]
    
    A_coeffs = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ]
    
    # For each block k
    for k in range(num_blocks):
        base_col = k * rank_per_block
        # B block indices: rows 2*k to 2*k+1 in B (which is 2x6)
        # B row 0: b11, b12 (indices 2*k, 2*k+1)
        # B row 1: b21, b22 (indices 2*k+2, 2*k+3)
        
        # Strassen coefficients for B (2x2)
        # B = [[e, f], [g, h]]
        # M1 = e+h -> [1, 0, 0, 1]
        # M2 = e   -> [1, 0, 0, 0]
        # M3 = f-h -> [0, 1, 0, -1]
        # M4 = g-e -> [-1, 0, 1, 0]
        # M5 = h   -> [0, 0, 0, 1]
        # M6 = e+f -> [1, 1, 0, 0]
        # M7 = g+h -> [0, 0, 1, 1]
        
        B_coeffs = [
            [1, 0, 0, 1], # M1
            [1, 0, 0, 0], # M2
            [0, 1, 0, -1],# M3
            [-1, 0, 1, 0],# M4
            [0, 0, 0, 1], # M5
            [1, 1, 0, 0], # M6
            [0, 0, 1, 1]  # M7
        ]
        
        # Output C block indices: rows 2*k to 2*k+1 in C (which is 6x2? No, C is 2x6)
        # Wait, W is (p*n, rank). p=6, n=2. So 12 rows.
        # The task says "output factor follows the official trace convention W[k*n+i,r]".
        # This implies C is indexed by [k, i] where k is col, i is row?
        # Standard matrix multiplication C = A B. C is n x p.
        # Index (i, k) in C corresponds to row i, col k.
        # The convention W[k*n+i, r] suggests the row index in W is k*n + i.
        # So for col k, row i, the index is k*n + i.
        # For k=0, i=0 -> 0. k=0, i=1 -> 1.
        # k=1, i=0 -> 2. k=1, i=1 -> 3.
        # ...
        # k=5, i=1 -> 11.
        
        # So for block k (which covers cols 2k and 2k+1):
        # C indices:
        # Col 2k, Row 0: (2k)*n + 0 = 4k
        # Col 2k, Row 1: (2k)*n + 1 = 4k + 1
        # Col 2k+1, Row 0: (2k+1)*n + 0 = 4k + 2
        # Col 2k+1, Row 1: (2k+1)*n + 1 = 4k + 3
        
        # C_coeffs for the 7 products
        # C11 = P1 + P4 - P5 + P7 -> [1, 0, 0, 1, -1, 0, 1]
        # C12 = P3 + P5           -> [0, 0, 1, 0, 1, 0, 0]
        # C21 = P2 + P4           -> [0, 1, 0, 1, 0, 0, 0]
        # C22 = P1 - P2 + P3 + P6 -> [1, -1, 1, 0, 0, 1, 0]
        
        C_coeffs = [
            [1, 0, 0, 1, -1, 0, 1], # C11 (Col 2k, Row 0)
            [0, 0, 1, 0, 1, 0, 0], # C12 (Col 2k+1, Row 0)
            [0, 1, 0, 1, 0, 0, 0], # C21 (Col 2k, Row 1)
            [1, -1, 1, 0, 0, 1, 0] # C22 (Col 2k+1, Row 1)
        ]
        
        # Fill U
        for r in range(rank_per_block):
            col = base_col + r
            for i in range(n*m):
                U[i][col] = A_coeffs[r][i]
                
        # Fill V
        # V rows correspond to B entries.
        # B is 2x6. Row j, Col k. Index j*p + k.
        # For block k, B cols are 2k and 2k+1.
        # B indices:
        # j=0, col=2k: 0*6 + 2k = 2k
        # j=0, col=2k+1: 0*6 + 2k+1 = 2k+1
        # j=1, col=2k: 1*6 + 2k = 6+2k
        # j=1, col=2k+1: 1*6 + 2k+1 = 6+2k+1
        
        # B_coeffs are for [e, f, g, h] which correspond to:
        # e = B[0, 2k]
        # f = B[0, 2k+1]
        # g = B[1, 2k]
        # h = B[1, 2k+1]
        
        B_indices = [2*k, 2*k+1, 6+2*k, 6+2*k+1]
        
        for r in range(rank_per_block):
            col = base_col + r
            for idx_in_block in range(4):
                v_row = B_indices[idx_in_block]
                V[v_row][col] = B_coeffs[r][idx_in_block]
                
        # Fill W
        # W rows correspond to C entries.
        # C indices as derived:
        # C11 (Col 2k, Row 0): 4k
        # C12 (Col 2k+1, Row 0): 4k+2
        # C21 (Col 2k, Row 1): 4k+1
        # C22 (Col 2k+1, Row 1): 4k+3
        
        # C_coeffs order:
        # 0: C11
        # 1: C12
        # 2: C21
        # 3: C22
        
        C_indices = [4*k, 4*k+2, 4*k+1, 4*k+3]
        
        for r in range(rank_per_block):
            col = base_col + r
            for idx_in_block in range(4):
                w_row = C_indices[idx_in_block]
                W[w_row][col] = C_coeffs[idx_in_block][r]
                
    return {"U": U, "V": V, "W": W}
