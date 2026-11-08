import numpy as np

def construct():
    n, m, p = 4, 4, 4
    rank = 49
    
    # U: (n*m, rank) = (16, 49)
    # V: (m*p, rank) = (16, 49)
    # W: (p*n, rank) = (16, 49)
    
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Helper to create a linear form vector
    # A index: i*4 + j
    # B index: j*4 + k
    
    # We will simulate the recursive Strassen.
    
    # Inner Strassen Coefficients
    # Returns (u_coeffs, v_coeffs, z_coeffs) for the 7 multiplications
    # u_coeffs: list of 7 vectors of length 16 (for A)
    # v_coeffs: list of 7 vectors of length 16 (for B)
    # z_coeffs: 2x2 matrix of 7-vectors, where z_coeffs[i][j][s] is coeff of m_s in z_ij
    
    def inner_strassen(x, y):
        # x, y are 2x2 matrices of linear forms (vectors of length 16)
        # x[i][j] is a vector
        # y[i][j] is a vector
        
        # m1 = (x00 + x11)(y00 + y11)
        u1 = [a + b for a, b in zip(x[0][0], x[1][1])]
        v1 = [a + b for a, b in zip(y[0][0], y[1][1])]
        
        # m2 = (x10 + x11)y00
        u2 = [a + b for a, b in zip(x[1][0], x[1][1])]
        v2 = y[0][0]
        
        # m3 = x00(y10 - y11)
        u3 = x[0][0]
        v3 = [a - b for a, b in zip(y[1][0], y[1][1])]
        
        # m4 = x11(y01 - y00)
        u4 = x[1][1]
        v4 = [a - b for a, b in zip(y[0][1], y[0][0])]
        
        # m5 = (x00 + x01)y11
        u5 = [a + b for a, b in zip(x[0][0], x[0][1])]
        v5 = y[1][1]
        
        # m6 = (x10 - x00)(y00 + y01)
        u6 = [a - b for a, b in zip(x[1][0], x[0][0])]
        v6 = [a + b for a, b in zip(y[0][0], y[0][1])]
        
        # m7 = (x01 - x11)(y10 + y11)
        u7 = [a - b for a, b in zip(x[0][1], x[1][1])]
        v7 = [a + b for a, b in zip(y[1][0], y[1][1])]
        
        u_forms = [u1, u2, u3, u4, u5, u6, u7]
        v_forms = [v1, v2, v3, v4, v5, v6, v7]
        
        # z_coeffs[i][j] is a vector of length 7
        # z00 = m1 + m4 - m5 + m7
        z00 = [0]*7
        z00[0] += 1
        z00[3] += 1
        z00[4] -= 1
        z00[6] += 1
        
        # z01 = m3 + m5
        z01 = [0]*7
        z01[2] += 1
        z01[4] += 1
        
        # z10 = m2 + m4
        z10 = [0]*7
        z10[1] += 1
        z10[3] += 1
        
        # z11 = m1 - m2 + m3 + m6
        z11 = [0]*7
        z11[0] += 1
        z11[1] -= 1
        z11[2] += 1
        z11[5] += 1
        
        z_coeffs = [[z00, z01], [z10, z11]]
        
        return u_forms, v_forms, z_coeffs

    # Outer Strassen Setup
    # A blocks: A00, A01, A10, A11
    # A00: indices 0..3 (0*4+0 to 0*4+3, 1*4+0 to 1*4+3) -> i in 0,1; j in 0,1
    # A01: i in 0,1; j in 2,3
    # A10: i in 2,3; j in 0,1
    # A11: i in 2,3; j in 2,3
    
    # We need to extract the linear forms for each block.
    # Since A is an input, its entries are basis vectors.
    # Let e_A[i*4+j] be the vector with 1 at i*4+j and 0 elsewhere.
    
    def get_block(A_basis, i_start, j_start):
        # Returns 2x2 matrix of vectors
        block = [[None]*2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                idx = (i_start + i) * 4 + (j_start + j)
                block[i][j] = A_basis[idx]
        return block

    # Initialize basis vectors for A and B
    A_basis = [np.zeros(16, dtype=int) for _ in range(16)]
    B_basis = [np.zeros(16, dtype=int) for _ in range(16)]
    for i in range(16):
        A_basis[i][i] = 1
        B_basis[i][i] = 1

    # Define blocks
    A00 = get_block(A_basis, 0, 0)
    A01 = get_block(A_basis, 0, 2)
    A10 = get_block(A_basis, 2, 0)
    A11 = get_block(A_basis, 2, 2)
    
    B00 = get_block(B_basis, 0, 0)
    B01 = get_block(B_basis, 0, 2)
    B10 = get_block(B_basis, 2, 0)
    B11 = get_block(B_basis, 2, 2)

    # Outer Forms X_r, Y_r
    # X_r and Y_r are 2x2 matrices of linear forms
    
    def add_blocks(B1, B2, sign=1):
        res = [[None]*2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                res[i][j] = [a + sign*b for a, b in zip(B1[i][j], B2[i][j])]
        return res

    def sub_blocks(B1, B2):
        return add_blocks(B1, B2, -1)

    X0 = add_blocks(A00, A11)
    Y0 = add_blocks(B00, B11)
    
    X1 = add_blocks(A10, A11)
    Y1 = B00
    
    X2 = A00
    Y2 = sub_blocks(B10, B11)
    
    X3 = A11
    Y3 = sub_blocks(B01, B00)
    
    X4 = add_blocks(A00, A01)
    Y4 = B11
    
    X5 = sub_blocks(A10, A00)
    Y5 = add_blocks(B00, B01)
    
    X6 = sub_blocks(A01, A11)
    Y6 = add_blocks(B10, B11)
    
    Xs = [X0, X1, X2, X3, X4, X5, X6]
    Ys = [Y0, Y1, Y2, Y3, Y4, Y5, Y6]
    
    # Process each outer multiplication
    # z_results[r] will be the 2x2 matrix of z_coeffs for that r
    z_results = []
    
    for r in range(7):
        u_forms, v_forms, z_coeffs = inner_strassen(Xs[r], Ys[r])
        
        # Fill U and V
        for s in range(7):
            col = r * 7 + s
            U[:, col] = u_forms[s]
            V[:, col] = v_forms[s]
            
        z_results.append(z_coeffs)
        
    # Fill W
    # Outer Reconstruction:
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # Map block indices (I, J) to global C indices
    # C[I*2 + i, J*2 + j]
    # W index: (J*2 + j) * 4 + (I*2 + i)
    
    outer_coeffs = {
        (0, 0): [(0, 1), (3, 1), (4, -1), (6, 1)],
        (0, 1): [(2, 1), (4, 1)],
        (1, 0): [(1, 1), (3, 1)],
        (1, 1): [(0, 1), (1, -1), (2, 1), (5, 1)]
    }
    
    for (I, J), contributions in outer_coeffs.items():
        for r, sign in contributions:
            # z_results[r] is 2x2 matrix of 7-vectors
            # z_results[r][i][j][s] is coeff of S_{r,s} in Z_r[i,j]
            for i in range(2):
                for j in range(2):
                    # Global indices in C
                    gi = I * 2 + i
                    gj = J * 2 + j
                    w_idx = gj * 4 + gi
                    
                    # Add contributions to W
                    for s in range(7):
                        col = r * 7 + s
                        W[w_idx, col] += sign * z_results[r][i][j][s]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
