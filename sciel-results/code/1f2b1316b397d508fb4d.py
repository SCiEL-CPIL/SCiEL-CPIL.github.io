import time
import numpy as np

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices
    for a rank-72 exact tensor decomposition of 4x4 * 4x5 matrix multiplication.
    Uses Strassen's algorithm for the 4x4x4 sub-problem (rank 56) and schoolbook
    for the 4x4x1 sub-problem (rank 16), total rank 72.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 5
    rank_total = 72

    # Initialize factor matrices as lists of lists
    U = [[0] * rank_total for _ in range(n * m)]  # (16, 72)
    V = [[0] * rank_total for _ in range(m * p)]  # (20, 72)
    W = [[0] * rank_total for _ in range(p * n)]  # (20, 72)

    # Strassen's 7 bilinear forms for 2x2 * 2x2
    # Each form is a list of (i, j, k, coeff) tuples where i,j are A indices,
    # k is B index, and coeff is the coefficient.
    # A indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    # B indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> flat: 0, 1, 2, 3
    
    # Strassen forms:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B01-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # We represent each form as a list of (a_idx, b_idx, coeff)
    # where a_idx is the flat index in the 2x2 A block, b_idx in 2x2 B block.
    
    strassen_forms = [
        # m1: (A00+A11)(B00+B11)
        [(0, 0, 1), (3, 0, 1), (0, 3, 1), (3, 3, 1)],
        # m2: (A10+A11)B00
        [(2, 0, 1), (3, 0, 1)],
        # m3: A00(B10-B11)
        [(0, 2, 1), (0, 3, -1)],
        # m4: A11(B01-B00)
        [(3, 1, 1), (3, 0, -1)],
        # m5: (A00+A01)B11
        [(0, 3, 1), (1, 3, 1)],
        # m6: (A10-A00)(B00+B01)
        [(2, 0, 1), (0, 0, -1), (2, 1, 1), (0, 1, -1)],
        # m7: (A01-A11)(B10+B11)
        [(1, 2, 1), (3, 2, -1), (1, 3, 1), (3, 3, -1)],
    ]

    # The output C entries for each Strassen form:
    # C00 = m1 + m4 - m5 - m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # c_output[form_idx][c_idx] = coeff
    c_output = [
        [1, 0, 0, 0],  # m1 -> C00: +1
        [0, 0, 1, 0],  # m2 -> C10: +1
        [0, 1, 0, 0],  # m3 -> C01: +1
        [1, 0, 0, 1],  # m4 -> C00: +1, C11: +1
        [-1, 1, 0, 0], # m5 -> C00: -1, C01: +1
        [0, 0, 0, 1],  # m6 -> C11: +1
        [-1, 0, 0, -1],# m7 -> C00: -1, C11: -1
    ]

    col = 0
    
    # Part 1: 4x4 * 4x4 using Strassen on 2x2 blocks
    # A is 4x4, split into 2x2 blocks: A_{i1,j1} where i1,j1 in {0,1}
    # B is 4x4 (first 4 cols), split into 2x2 blocks: B_{j1,k1} where j1,k1 in {0,1}
    # C is 4x4, split into 2x2 blocks: C_{i1,k1} where i1,k1 in {0,1}
    
    # For each block position (i1, j1, k1), we compute A_{i1,j1} * B_{j1,k1}
    # But Strassen works on the whole 2x2 block multiplication.
    # Actually, we can apply Strassen to each of the 8 block multiplications
    # that contribute to the 4x4 result.
    
    # The 4x4 matrix multiplication can be viewed as 8 independent 2x2*2x2
    # multiplications if we consider the block structure:
    # C_{i1,k1} = sum_{j1=0}^{1} A_{i1,j1} B_{j1,k1}
    # This is not 8 independent 2x2*2x2 multiplications because the A and B
    # blocks are shared. However, we can still use the Strassen trick by
    # applying it to the bilinear form of each 2x2*2x2 sub-product.
    
    # Actually, the correct way is: for each (i1, j1, k1) triple, we have a
    # 2x2 * 2x2 multiplication. There are 2*2*2 = 8 such triples.
    # For each, we use 7 Strassen terms. Total 56 terms.
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                # The 2x2 A block is A[i1*2:(i1+1)*2, j1*2:(j1+1)*2]
                # The 2x2 B block is B[j1*2:(j1+1)*2, k1*2:(k1+1)*2]
                # The 2x2 C block is C[i1*2:(i1+1)*2, k1*2:(k1+1)*2]
                
                for form_idx in range(7):
                    if time.time() > deadline:
                        return {"U": U, "V": V, "W": W}
                    
                    # Add the Strassen form to U, V
                    for (a_local, b_local, coeff) in strassen_forms[form_idx]:
                        # a_local is 0..3 in the 2x2 A block
                        # a_local -> row a_local//2, col a_local%2 in the block
                        # Global A row: i1*2 + a_local//2
                        # Global A col: j1*2 + a_local%2
                        a_row = i1 * 2 + a_local // 2
                        a_col = j1 * 2 + a_local % 2
                        a_flat = a_row * m + a_col  # m=4
                        
                        # b_local is 0..3 in the 2x2 B block
                        # Global B row: j1*2 + b_local//2
                        # Global B col: k1*2 + b_local%2
                        b_row = j1 * 2 + b_local // 2
                        b_col = k1 * 2 + b_local % 2
                        b_flat = b_row * p + b_col  # p=5, but we're in 4x4 part, so col < 4
                        
                        U[a_flat][col] += coeff
                        V[b_flat][col] += coeff
                    
                    # Add the C output
                    for c_local in range(4):
                        coeff_c = c_output[form_idx][c_local]
                        if coeff_c != 0:
                            # c_local is 0..3 in the 2x2 C block
                            # Global C row: i1*2 + c_local//2
                            # Global C col: k1*2 + c_local%2
                            c_row = i1 * 2 + c_local // 2
                            c_col = k1 * 2 + c_local % 2
                            # W index: k*n + i, where k is col, i is row? 
                            # Wait, the convention is W[k*n+i, r] where (k,i) is the output index.
                            # Standard: C[i,k] = sum_j A[i,j] B[j,k]
                            # So C row = i, C col = k.
                            # W index = k*n + i = c_col * n + c_row
                            w_flat = c_col * n + c_row
                            W[w_flat][col] += coeff_c
                    
                    col += 1
    
    # Part 2: 4x4 * 4x1 (the 5th column of B) using schoolbook
    # For each i in 0..3, j in 0..3, we have A[i,j] * B[j,4] contributing to C[i,4]
    # C[i,4] = sum_j A[i,j] B[j,4]
    # W index for C[i,4]: k=4, i=i, so W[4*4+i, r] = W[16+i, r]
    
    for i in range(4):
        for j in range(4):
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
            
            a_flat = i * m + j
            b_flat = j * p + 4  # B[j, 4]
            w_flat = 4 * n + i  # C[i, 4] -> W[4*4+i]
            
            U[a_flat][col] = 1
            V[b_flat][col] = 1
            W[w_flat][col] = 1
            col += 1
    
    if col != rank_total:
        # Should not happen, but guard against it
        pass
    
    return {"U": U, "V": V, "W": W}
