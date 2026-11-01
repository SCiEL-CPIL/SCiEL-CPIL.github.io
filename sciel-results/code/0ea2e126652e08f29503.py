import numpy as np

DIMENSIONS = [2, 2, 14]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=14
    
    # Strassen's 7 intermediate products for 2x2 matrix multiplication
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A11)*(B10+B11)
    # M7 = (A01-A10)*(B00+B10)
    
    # For each 2x2 block of B (there are p/2 = 7 blocks), we apply Strassen
    # U has shape (n*m, rank) = (4, 49)
    # V has shape (m*p, rank) = (28, 49)  
    # W has shape (p*n, rank) = (28, 49)
    
    rank = 7 * 7  # 7 blocks * 7 Strassen terms = 49
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen coefficients: for each term r in 0..6, define U, V, W contributions
    # U indices: A00=0, A01=1, A10=2, A11=3
    # V indices for one 2x2 block: B00=0, B01=1, B10=2, B11=3 (local)
    # W indices for one 2x2 block: C00=0, C01=1, C10=2, C11=3 (local)
    
    # Strassen term definitions: (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (A_index, coeff)
    # V_coeffs: list of (B_local_index, coeff)
    # W_coeffs: list of (C_local_index, coeff)
    
    strassen_terms = [
        # M1: (A00+A11)*(B00+B11) -> contributes to C00, C11
        ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
        # M2: (A10+A11)*B00 -> contributes to C10, C11
        ([ (2, 1), (3, 1) ], [ (0, 1) ], [ (2, 1), (3, -1) ]),
        # M3: A00*(B01-B11) -> contributes to C01, C11
        ([ (0, 1) ], [ (1, 1), (3, -1) ], [ (1, 1), (3, 1) ]),
        # M4: A11*(B10-B00) -> contributes to C10, C11
        ([ (3, 1) ], [ (2, 1), (0, -1) ], [ (2, 1), (3, 1) ]),
        # M5: (A00+A10)*B11 -> contributes to C00, C01
        ([ (0, 1), (2, 1) ], [ (3, 1) ], [ (0, -1), (1, 1) ]),
        # M6: (A01-A11)*(B10+B11) -> contributes to C11
        ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ], [ (3, 1) ]),
        # M7: (A01-A10)*(B00+B10) -> contributes to C00, C10
        ([ (1, 1), (2, -1) ], [ (0, 1), (2, 1) ], [ (0, 1), (2, 1) ]),
    ]
    
    num_blocks = p // 2  # 7 blocks
    
    for block in range(num_blocks):
        for r in range(7):
            col = block * 7 + r
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[r]
            
            # Set U entries (same for all blocks since A is shared)
            for a_idx, coeff in u_coeffs:
                U[a_idx, col] += coeff
            
            # Set V entries (local B indices mapped to global)
            # Global B index for block b, local index l: b*2 + l (since m=2, each block has 2 columns)
            for b_local, coeff in v_coeffs:
                global_v_idx = block * 2 + b_local
                V[global_v_idx, col] += coeff
            
            # Set W entries (local C indices mapped to global)
            # Global C index: k*n + i where k is column index in B (0..13), i is row index in A (0..1)
            # For block b, local C index l corresponds to:
            # l=0 -> C00: k=2*block, i=0 -> global = 2*block * n + 0 = 2*block*2 + 0 = 4*block
            # l=1 -> C01: k=2*block+1, i=0 -> global = (2*block+1)*2 + 0 = 4*block + 2
            # l=2 -> C10: k=2*block, i=1 -> global = 2*block*2 + 1 = 4*block + 1
            # l=3 -> C11: k=2*block+1, i=1 -> global = (2*block+1)*2 + 1 = 4*block + 3
            w_index_map = {0: 4*block + 0, 1: 4*block + 2, 2: 4*block + 1, 3: 4*block + 3}
            for c_local, coeff in w_coeffs:
                global_w_idx = w_index_map[c_local]
                W[global_w_idx, col] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
