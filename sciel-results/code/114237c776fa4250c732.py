import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define sub-blocks
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x6, split into 2x2 blocks: B11, B12, B13, B21, B22, B23
    # C is 4x6, split into 2x2 blocks: C11, C12, C13, C21, C22, C23
    
    # Indices for A (row, col)
    A_blocks = {
        'A11': [(0,0), (0,1), (1,0), (1,1)],
        'A12': [(0,2), (0,3), (1,2), (1,3)],
        'A21': [(2,0), (2,1), (3,0), (3,1)],
        'A22': [(2,2), (2,3), (3,2), (3,3)]
    }
    
    # Indices for B (row, col)
    B_blocks = {
        'B11': [(0,0), (0,1), (1,0), (1,1)],
        'B12': [(0,2), (0,3), (1,2), (1,3)],
        'B13': [(0,4), (0,5), (1,4), (1,5)],
        'B21': [(2,0), (2,1), (3,0), (3,1)],
        'B22': [(2,2), (2,3), (3,2), (3,3)],
        'B23': [(2,4), (2,5), (3,4), (3,5)]
    }
    
    # Indices for C (row, col)
    C_blocks = {
        'C11': [(0,0), (0,1), (1,0), (1,1)],
        'C12': [(0,2), (0,3), (1,2), (1,3)],
        'C13': [(0,4), (0,5), (1,4), (1,5)],
        'C21': [(2,0), (2,1), (3,0), (3,1)],
        'C22': [(2,2), (2,3), (3,2), (3,3)],
        'C23': [(2,4), (2,5), (3,4), (3,5)]
    }
    
    # Sub-multiplications: (A_block, B_block, C_block)
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C13 = A11 B13 + A12 B23
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    # C23 = A21 B13 + A22 B23
    
    # We need to decompose each product A_i B_j into C_k
    # But C_k is a sum of two products.
    # So we treat each product A_i B_j as a separate tensor that contributes to C_k.
    # The total tensor is the sum of these 12 product tensors.
    
    products = [
        ('A11', 'B11', 'C11'),
        ('A12', 'B21', 'C11'),
        ('A11', 'B12', 'C12'),
        ('A12', 'B22', 'C12'),
        ('A11', 'B13', 'C13'),
        ('A12', 'B23', 'C13'),
        ('A21', 'B11', 'C21'),
        ('A22', 'B21', 'C21'),
        ('A21', 'B12', 'C22'),
        ('A22', 'B22', 'C22'),
        ('A21', 'B13', 'C23'),
        ('A22', 'B23', 'C23')
    ]
    
    # Strassen factors for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 - M6
    
    # We define the linear forms for A and B for each M_k
    # and the coefficients for C.
    
    # A indices in local 2x2: 00, 01, 10, 11
    # B indices in local 2x2: 00, 01, 10, 11
    # C indices in local 2x2: 00, 01, 10, 11
    
    strassen_forms = [
        # (A_coeffs, B_coeffs, C_coeffs)
        # M1
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        # M2
        ({2:1, 3:1}, {0:1}, {2:1}), # C10 gets M2. Local C index 2 is (1,0).
        # M3
        ({0:1}, {1:1, 3:-1}, {1:1}), # C01 gets M3. Local C index 1 is (0,1).
        # M4
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}), # C00 gets M4, C10 gets M4.
        # M5
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}), # C00 gets -M5, C01 gets M5.
        # M6
        ({2:1, 0:-1}, {0:1, 1:1}, {3:-1}), # C11 gets -M6. Local C index 3 is (1,1).
        # M7
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})  # C00 gets M7.
    ]
    
    col = 0
    for a_name, b_name, c_name in products:
        a_indices = A_blocks[a_name]
        b_indices = B_blocks[b_name]
        c_indices = C_blocks[c_name]
        
        # Map local indices (0,1,2,3) to global indices
        # Local 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
        local_to_global_A = {
            0: a_indices[0],
            1: a_indices[1],
            2: a_indices[2],
            3: a_indices[3]
        }
        local_to_global_B = {
            0: b_indices[0],
            1: b_indices[1],
            2: b_indices[2],
            3: b_indices[3]
        }
        local_to_global_C = {
            0: c_indices[0],
            1: c_indices[1],
            2: c_indices[2],
            3: c_indices[3]
        }
        
        for k in range(7):
            a_coeffs, b_coeffs, c_coeffs = strassen_forms[k]
            
            # Update U
            for local_idx, coeff in a_coeffs.items():
                r, c = local_to_global_A[local_idx]
                U[r * m + c][col] = coeff
            
            # Update V
            for local_idx, coeff in b_coeffs.items():
                r, c = local_to_global_B[local_idx]
                V[r * p + c][col] = coeff
            
            # Update W
            # W index is k*n + i, where k is col of C, i is row of C.
            # Wait, the task says W[k*n+i, r].
            # C is p x n. So k is 0..p-1, i is 0..n-1.
            # C indices are (row, col). row is 0..n-1, col is 0..p-1.
            # So W[col_C * n + row_C, rank_idx] = coeff.
            for local_idx, coeff in c_coeffs.items():
                r, c = local_to_global_C[local_idx]
                # r is row of C (0..3), c is col of C (0..5)
                W[c * n + r][col] = coeff
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
