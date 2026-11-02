import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # R = 76
    
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a scalar multiplication column
    # a_idx: list of (index, coeff) for A
    # b_idx: list of (index, coeff) for B
    # c_idx: list of (index, coeff) for C
    def add_col(a_idx, b_idx, c_idx):
        nonlocal col
        for idx, coeff in a_idx:
            U[idx][col] += coeff
        for idx, coeff in b_idx:
            V[idx][col] += coeff
        for idx, coeff in c_idx:
            W[idx][col] += coeff
        col += 1
        
    # Index helpers
    def a_idx(r, c): return r * 4 + c
    def b_idx(r, c): return r * 7 + c
    def c_idx(r, c): return c * 4 + r # W[k*n + i] where k is col of C, i is row of C.
                                      # C is 4x7. C[i][k]. W index is k*4 + i.
                                      # In the problem, C[i][k] = sum A[i][j] B[j][k].
                                      # So C row i, col k.
                                      # W index = k * n + i = k * 4 + i.
                                      # My c_idx(r, c) above: r is row of C (i), c is col of C (k).
                                      # So c_idx(i, k) = k * 4 + i. Correct.

    # Strassen Part: A (4x4) * B_1 (4x4) -> C_1 (4x4)
    # B_1 corresponds to columns 0-3 of B.
    
    # Define block indices
    # A blocks are 2x2
    # A11: r 0-1, c 0-1
    # A12: r 0-1, c 2-3
    # A21: r 2-3, c 0-1
    # A22: r 2-3, c 2-3
    
    # B blocks are 2x2 (for B_1)
    # B11: r 0-1, c 0-1
    # B12: r 0-1, c 2-3
    # B21: r 2-3, c 0-1
    # B22: r 2-3, c 2-3
    
    # C blocks are 2x2
    # C11: r 0-1, c 0-1
    # C12: r 0-1, c 2-3
    # C21: r 2-3, c 0-1
    # C22: r 2-3, c 2-3
    
    # Strassen Terms
    # Each term is a product of two 2x2 matrices.
    # We expand each term into 4 scalar multiplications (schoolbook).
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Reconstruction
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 - M6
    
    # We will generate columns for each scalar multiplication in each M_k.
    # For a term M_k, which is a 2x2 matrix product X*Y,
    # M_k[i][j] = sum_l X[i][l] Y[l][j]
    # We add a column for each (i, j, l) combination.
    
    def process_strassen_term(a_blocks, b_blocks, c_contributions):
        """
        a_blocks: list of (block_name, coeff)
        b_blocks: list of (block_name, coeff)
        c_contributions: dict mapping C_block_name to coeff
        """
        # Get the actual indices for the blocks
        # We need to expand the product of the linear combinations.
        # X = sum ca * A_block
        # Y = sum cb * B_block
        # M = X Y
        # M[i][j] = sum_l (sum ca A_block[i][l]) (sum cb B_block[l][j])
        #         = sum_l sum ca sum cb A_block[i][l] B_block[l][j]
        
        # We iterate over i, j, l (0..1)
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    a_idx_list = []
                    b_idx_list = []
                    c_idx_list = []
                    
                    for block_name, ca in a_blocks:
                        # Get the index of A_block[i][l]
                        # Block names: 'A11', 'A12', 'A21', 'A22'
                        br, bc = divmod(i, 2) # i is 0 or 1
                        # Actually, block is 2x2. i is row in block, l is col in block.
                        # Global row: br_offset + i, Global col: bc_offset + l
                        if block_name == 'A11':
                            gr, gc = 0 + i, 0 + l
                        elif block_name == 'A12':
                            gr, gc = 0 + i, 2 + l
                        elif block_name == 'A21':
                            gr, gc = 2 + i, 0 + l
                        elif block_name == 'A22':
                            gr, gc = 2 + i, 2 + l
                        else:
                            continue
                        a_idx_list.append((a_idx(gr, gc), ca))
                        
                    for block_name, cb in b_blocks:
                        # Get the index of B_block[l][j]
                        # l is row in block, j is col in block.
                        if block_name == 'B11':
                            gr, gc = 0 + l, 0 + j
                        elif block_name == 'B12':
                            gr, gc = 0 + l, 2 + j
                        elif block_name == 'B21':
                            gr, gc = 2 + l, 0 + j
                        elif block_name == 'B22':
                            gr, gc = 2 + l, 2 + j
                        else:
                            continue
                        b_idx_list.append((b_idx(gr, gc), cb))
                        
                    # C contributions
                    # M[i][j] contributes to C blocks.
                    # The position (i, j) in M corresponds to a specific position in the C block.
                    # M is 2x2.
                    # C11 is M1 - M4 + M5 - M7.
                    # The element M_k[i][j] contributes to C_block[i][j] with the coefficient from c_contributions.
                    for c_block_name, cc in c_contributions.items():
                        if c_block_name == 'C11':
                            gr, gc = 0 + i, 0 + j
                        elif c_block_name == 'C12':
                            gr, gc = 0 + i, 2 + j
                        elif c_block_name == 'C21':
                            gr, gc = 2 + i, 0 + j
                        elif c_block_name == 'C22':
                            gr, gc = 2 + i, 2 + j
                        else:
                            continue
                        c_idx_list.append((c_idx(gr, gc), cc))
                        
                    add_col(a_idx_list, b_idx_list, c_idx_list)
                    
    # Define the terms
    # M1
    process_strassen_term(
        [('A11', 1), ('A22', 1)],
        [('B11', 1), ('B22', 1)],
        {'C11': 1, 'C22': 1}
    )
    # M2
    process_strassen_term(
        [('A21', 1), ('A22', 1)],
        [('B11', 1)],
        {'C21': 1, 'C22': 1}
    )
    # M3
    process_strassen_term(
        [('A11', 1)],
        [('B12', 1), ('B22', -1)],
        {'C12': 1, 'C22': -1}
    )
    # M4
    process_strassen_term(
        [('A22', 1)],
        [('B21', 1), ('B11', -1)],
        {'C11': -1, 'C21': 1}
    )
    # M5
    process_strassen_term(
        [('A11', 1), ('A12', 1)],
        [('B22', 1)],
        {'C11': 1}
    )
    # M6
    process_strassen_term(
        [('A21', 1), ('A11', -1)],
        [('B11', 1), ('B12', 1)],
        {'C22': -1}
    )
    # M7
    process_strassen_term(
        [('A12', 1), ('A22', -1)],
        [('B21', 1), ('B22', 1)],
        {'C11': -1, 'C12': 1}
    )
    
    # Schoolbook Part: A (4x4) * B_2 (4x3) -> C_2 (4x3)
    # B_2 corresponds to columns 4-6 of B.
    # C_2 corresponds to columns 4-6 of C.
    
    # C13 (rows 0-1, cols 4-6) = A11 B13 + A12 B23
    # C23 (rows 2-3, cols 4-6) = A21 B13 + A22 B23
    
    # B13: rows 0-1, cols 4-6 (2x3)
    # B23: rows 2-3, cols 4-6 (2x3)
    
    # For C13[i][k] (i in 0-1, k in 4-6):
    # C13[i][k] = sum_l (A11[i][l] B13[l][k-4] + A12[i][l] B23[l][k-4])
    
    for i in range(2): # Row in C13
        for k in range(3): # Col in C13 (0-2, maps to 4-6 in B/C)
            for l in range(2): # Sum index
                # Term 1: A11[i][l] * B13[l][k]
                a_idx_list = [(a_idx(0 + i, 0 + l), 1)]
                b_idx_list = [(b_idx(0 + l, 4 + k), 1)]
                c_idx_list = [(c_idx(0 + i, 4 + k), 1)]
                add_col(a_idx_list, b_idx_list, c_idx_list)
                
                # Term 2: A12[i][l] * B23[l][k]
                a_idx_list = [(a_idx(0 + i, 2 + l), 1)]
                b_idx_list = [(b_idx(2 + l, 4 + k), 1)]
                c_idx_list = [(c_idx(0 + i, 4 + k), 1)]
                add_col(a_idx_list, b_idx_list, c_idx_list)
                
    for i in range(2): # Row in C23
        for k in range(3): # Col in C23 (0-2, maps to 4-6 in B/C)
            for l in range(2): # Sum index
                # Term 1: A21[i][l] * B13[l][k]
                a_idx_list = [(a_idx(2 + i, 0 + l), 1)]
                b_idx_list = [(b_idx(0 + l, 4 + k), 1)]
                c_idx_list = [(c_idx(2 + i, 4 + k), 1)]
                add_col(a_idx_list, b_idx_list, c_idx_list)
                
                # Term 2: A22[i][l] * B23[l][k]
                a_idx_list = [(a_idx(2 + i, 2 + l), 1)]
                b_idx_list = [(b_idx(2 + l, 4 + k), 1)]
                c_idx_list = [(c_idx(2 + i, 4 + k), 1)]
                add_col(a_idx_list, b_idx_list, c_idx_list)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
