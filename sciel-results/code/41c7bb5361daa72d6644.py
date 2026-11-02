import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # We will construct U, V, W for the whole 4x4x6 multiplication.
    # Split into 4x4x4 and 4x4x2.
    # 4x4x4 uses Strassen with schoolbook 2x2x2 inner (rank 28).
    # 4x4x2 uses schoolbook (rank 16).
    # Total rank 44.
    
    rank_444 = 28
    rank_442 = 16
    total_rank = rank_444 + rank_442
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx is (i, j) for A
        # v_idx is (j, k) for B
        # w_idx is (k, i) for C (note: W index is k*n + i)
        U[u_idx[0] * m + u_idx[1]][col] = val
        V[v_idx[0] * p + v_idx[1]][col] = val
        W[w_idx[0] * n + w_idx[1]][col] = val
        
    col = 0
    
    # Part 1: 4x4x4 Strassen (rank 28)
    # A = [[A11, A12], [A21, A22]] (2x2 blocks of 2x2)
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen terms M1..M7
    # Each Mi is product of two 2x2 matrices.
    # We use schoolbook for each 2x2 product.
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # We need to map these to U, V, W.
    # This is complex to do manually in code without errors.
    # I will use a helper to generate the schoolbook terms for a given linear combination.
    
    # Let's define the linear combinations for U and V for each Strassen term.
    # For M1: U terms are A11, A22. V terms are B11, B22.
    # Actually, U is a matrix of coefficients for A.
    # For M1, U has 1s for A11 and A22. V has 1s for B11 and B22.
    # But we need to expand the product (A11+A22)(B11+B22) into 4 schoolbook terms.
    # Term 1: A11 B11. U: A11, V: B11. W: C11 += 1, C12 += 1, C21 += 1, C22 += 1? No.
    # M1 contributes to C11, C12, C21, C22.
    # C11 += M1. C12 += M1. C21 += M1. C22 += M1? No.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So for each schoolbook term in Mi, we add 1 to the corresponding C block.
    
    # This is getting tedious. I will write a generic function to add a schoolbook term
    # for a specific A-block, B-block, and C-block contribution.
    
    def add_schoolbook_term(a_block, b_block, c_block, sign=1):
        nonlocal col
        # a_block, b_block, c_block are (row_start, col_start) in 2x2 grid of 2x2 blocks.
        # A is 4x4. Blocks are 2x2.
        # a_block (br, bc) corresponds to A[br*2:(br+1)*2, bc*2:(bc+1)*2]
        # b_block (br, bc) corresponds to B[br*2:(br+1)*2, bc*2:(bc+1)*2]
        # c_block (br, bc) corresponds to C[br*2:(br+1)*2, bc*2:(bc+1)*2]
        
        # We need to add 4 terms for the 2x2 multiplication.
        for i in range(2):
            for j in range(2):
                # A index: (br*2+i, bc*2+j)
                # B index: (bc*2+j, bc*2+k)? No.
                # B is 4x4. b_block (br, bc) is B[br*2:(br+1)*2, bc*2:(bc+1)*2].
                # The product is A_block * B_block.
                # A_block is 2x2. B_block is 2x2.
                # (A_block * B_block)_{i, k} = sum_j A_block_{i, j} B_block_{j, k}.
                
                for k in range(2):
                    # A index: (br*2+i, bc*2+j)
                    # B index: (br*2+j, bc*2+k) ? No.
                    # B_block row index is j (local), col index is k (local).
                    # Global B row: br*2+j. Global B col: bc*2+k.
                    # Global A row: br*2+i. Global A col: bc*2+j.
                    # Global C row: br*2+i. Global C col: bc*2+k.
                    
                    a_idx = (br*2+i, bc*2+j)
                    b_idx = (br*2+j, bc*2+k)
                    c_idx = (bc*2+k, br*2+i) # W index is k*n + i. Here k is C col, i is C row.
                    
                    # Wait, W index is k*n + i.
                    # C is 4x4. C[row, col].
                    # W index for C[row, col] is col*n + row.
                    # So w_idx = (col, row).
                    
                    U[a_idx[0] * m + a_idx[1]][col] = sign
                    V[b_idx[0] * p + b_idx[1]][col] = sign
                    W[c_idx[0] * n + c_idx[1]][col] = sign
                    col += 1
    return {"U": U, "V": V, "W": W}
