import numpy as np

def construct():
    n, m, p = 4, 5, 5
    rank = 85
    
    # Initialize U, V, W as numpy arrays for efficient assignment
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen factors for 2x2x2
    # u, v, w are 4x7 matrices
    s_u = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    s_v = np.array([
        [1, 0, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 1, -1, 0, 1, 0, 1]
    ])
    s_w = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Helper to add a 4x4x4 Strassen block
    # This function adds the terms for a 4x4x4 multiplication located at
    # rows i0:i0+4, cols j0:j0+4 of A
    # rows j0:j0+4, cols k0:k0+4 of B
    # rows k0:k0+4, cols i0:i0+4 of C
    def add_strassen_4x4(i0, j0, k0, start_col):
        # The 4x4x4 tensor is composed of 7 2x2x2 tensors
        # We iterate through the 7 Strassen terms
        for t in range(7):
            col = start_col + t
            # u_vec, v_vec, w_vec are 4-element vectors (2x2 matrices flattened)
            u_vec = s_u[:, t]
            v_vec = s_v[:, t]
            w_vec = s_w[:, t]
            
            # Unpack 2x2 matrices
            # u_vec: [u00, u01, u10, u11]
            # Note: The Strassen factors above are defined for 2x2 matrices.
            # We need to map them to the 4x4 block.
            
            # A block is 4x4. It consists of four 2x2 sub-blocks.
            # A_{00}, A_{01}, A_{10}, A_{11}
            # Indices in A:
            # A_{00}: rows i0..i0+1, cols j0..j0+1
            # A_{01}: rows i0..i0+1, cols j0+2..j0+3
            # A_{10}: rows i0+2..i0+3, cols j0..j0+1
            # A_{11}: rows i0+2..i0+3, cols j0+2..j0+3
            
            # The Strassen decomposition operates on the 2x2 block matrices.
            # M1 = (A00+A11)(B00+B11)
            # ...
            # Each M is a 2x2 matrix multiplication.
            # So for each Strassen term t, we have a 2x2x2 multiplication.
            # This 2x2x2 multiplication has rank 1? No, rank 7 for the whole 4x4x4.
            # Wait, the 4x4x4 decomposition uses 7 terms, each of which is a 2x2x2 multiplication?
            # No, the 4x4x4 decomposition *is* the 7 Strassen terms.
            # Each term t corresponds to a product of two 2x2 matrices (from A and B blocks)
            # resulting in a 2x2 matrix (part of C block).
            # So for term t, we add 1 rank-1 tensor? No.
            # The rank of the 4x4x4 tensor is 7.
            # This means we add 7 columns to U, V, W.
            # For each column (term t), U, V, W have non-zero entries corresponding to the 2x2 matrices.
            
            # For term t:
            # U part: The 2x2 matrix from A.
            # V part: The 2x2 matrix from B.
            # W part: The 2x2 matrix from C.
            
            # The entries in U for this term are the elements of the A-matrix for term t.
            # s_u[:, t] gives the coefficients for the 4 elements of the 2x2 A-matrix.
            # We need to place these into the 4x4 A-block.
            
            # Mapping 2x2 indices (lr, lc) to 4x4 indices
            # The 2x2 matrices are:
            # A00, A01, A10, A11
            # s_u[0] -> A00[0,0]
            # s_u[1] -> A00[0,1]
            # s_u[2] -> A00[1,0]
            # s_u[3] -> A00[1,1]
            # Wait, my s_u definition above was for a single 2x2 matrix?
            # Yes, s_u is 4x7. Each column is a 2x2 matrix flattened.
            # But the Strassen algorithm uses *combinations* of the 4 sub-blocks.
            # e.g., M1 = (A00+A11)(B00+B11).
            # The A-part of M1 is A00+A11.
            # This is a 2x2 matrix.
            # So for term 0, the A-matrix is A00+A11.
            # The V-matrix is B00+B11.
            # The W-matrix is C00+C11.
            
            # My s_u, s_v, s_w above are the factors for a *single* 2x2x2 multiplication.
            # To use them for 4x4x4, I need to define the 7 "super-terms".
            # This is what the original program did with `terms`.
            
            # I will replicate the logic of the original program but using NumPy.
            
            pass # Placeholder, logic below

    # Re-implementing the term addition with NumPy
    terms = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict mapping (br, bc) to coeff
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    for t_idx, (a_coeff, b_coeff, c_coeff) in enumerate(terms):
        start_col = t_idx * 7
        for inner_k in range(7):
            col_idx = start_col + inner_k
            
            u_vec = s_u[:, inner_k]
            v_vec = s_v[:, inner_k]
            w_vec = s_w[:, inner_k]
            
            # Add U contributions
            for (a_br, a_bc), a_c in a_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        u_val = u_vec[local_idx]
                        if u_val != 0:
                            global_i = 2 * a_br + lr
                            global_j = 2 * a_bc + lc
                            global_u_row = global_i * 5 + global_j
                            U[global_u_row, col_idx] += a_c * u_val
            
            # Add V contributions
            for (b_br, b_bc), b_c in b_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        v_val = v_vec[local_idx]
                        if v_val != 0:
                            global_j = 2 * b_br + lr
                            global_k = 2 * b_bc + lc
                            global_v_row = global_j * 5 + global_k
                            V[global_v_row, col_idx] += b_c * v_val
            
            # Add W contributions
            for (c_br, c_bc), c_c in c_coeff.items():
                for lr in range(2):
                    for lc in range(2):
                        local_idx = lr * 2 + lc
                        w_val = w_vec[local_idx]
                        if w_val != 0:
                            global_k = 2 * c_bc + lc
                            global_i = 2 * c_br + lr
                            global_w_row = global_k * 4 + global_i
                            W[global_w_row, col_idx] += c_c * w_val

    # Add naive terms for strips
    col = 49
    for i in range(4):
        for j in range(4):
            U[i*5 + j, col] = 1
            V[j*5 + 4, col] = 1
            W[4*4 + i, col] = 1
            col += 1
            
    for i in range(4):
        for k in range(4):
            U[i*5 + 4, col] = 1
            V[4*5 + k, col] = 1
            W[k*4 + i, col] = 1
            col += 1
            
    for i in range(4):
        U[i*5 + 4, col] = 1
        V[4*5 + 4, col] = 1
        W[4*4 + i, col] = 1
        col += 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
