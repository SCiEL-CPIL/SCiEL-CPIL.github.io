import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to get block indices
    # A[i, j] is at index i*4 + j
    # Block A_{pq} contains A[2p+r, 2q+s]
    
    # Define the 16-vectors for A and B blocks
    # A_blocks[p, q] is a list of 4 local indices? No, just use global indices.
    # We'll define vectors for specific linear combinations.
    
    def get_vec_A(p, q):
        """Vector for block A_{pq}"""
        vec = [0] * 16
        for r in range(2):
            for s in range(2):
                vec[(2*p + r) * 4 + (2*q + s)] = 1
        return vec

    def get_vec_B(p, q):
        """Vector for block B_{pq}"""
        vec = [0] * 16
        for r in range(2):
            for s in range(2):
                vec[(2*p + r) * 4 + (2*q + s)] = 1
        return vec

    def add_vec(a, b, sign=1):
        return [ai + sign * bi for ai, bi in zip(a, b)]

    # Define X and Y inputs for each M_k
    # X[k] is a list of 4 vectors (x00, x01, x10, x11)
    # Y[k] is a list of 4 vectors (y00, y01, y10, y11)
    
    A00 = get_vec_A(0, 0)
    A01 = get_vec_A(0, 1)
    A10 = get_vec_A(1, 0)
    A11 = get_vec_A(1, 1)
    
    B00 = get_vec_B(0, 0)
    B01 = get_vec_B(0, 1)
    B10 = get_vec_B(1, 0)
    B11 = get_vec_B(1, 1)
    
    X_inputs = []
    Y_inputs = []
    
    # M_0: X = A00 + A11, Y = B00 + B11
    x = add_vec(A00, A11)
    y = add_vec(B00, B11)
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_1: X = A10 + A11, Y = B00
    x = add_vec(A10, A11)
    y = B00
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_2: X = A00, Y = B01 - B11
    x = A00
    y = add_vec(B01, B11, -1)
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_3: X = A11, Y = B10 - B00
    x = A11
    y = add_vec(B10, B00, -1)
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_4: X = A00 + A01, Y = B11
    x = add_vec(A00, A01)
    y = B11
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_5: X = A10 - A00, Y = B00 + B01
    x = add_vec(A10, A00, -1)
    y = add_vec(B00, B01)
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # M_6: X = A01 - A11, Y = B10 + B11
    x = add_vec(A01, A11, -1)
    y = add_vec(B10, B11)
    X_inputs.append([x, x, x, x])
    Y_inputs.append([y, y, y, y])
    
    # Strassen coefficients for U and V
    # Indices: 0:00, 1:01, 2:10, 3:11
    strassen_X = [
        [1, 0, 0, 1], # S0
        [0, 0, 1, 1], # S1
        [1, 0, 0, 0], # S2
        [0, 0, 0, 1], # S3
        [1, 1, 0, 0], # S4
        [0, 0, 1, -1],# S5
        [0, 1, 0, -1] # S6
    ]
    strassen_Y = [
        [1, 0, 0, 1], # S0
        [1, 0, 0, 0], # S1
        [0, 1, 0, -1],# S2
        [0, 0, 1, -1],# S3
        [0, 0, 0, 1], # S4
        [1, 1, 0, 0], # S5
        [0, 0, 1, 1]  # S6
    ]
    
    # Strassen coefficients for Z reconstruction
    # Z00, Z01, Z10, Z11 in terms of S0..S6
    strassen_Z = [
        [1, 0, 0, 1, -1, 0, 1], # Z00
        [0, 0, 1, 0, 1, 0, 0],  # Z01
        [0, 1, 0, 1, 0, 0, 0],  # Z10
        [1, -1, 1, 0, 0, 1, 0]  # Z11
    ]
    
    # C reconstruction from M_k
    # C_pq = sum_k alpha[p][q][k] M_k
    # p, q in 0..1
    alpha = [
        [ [1, 0, 0, 1, -1, 0, 1],  # C00
          [0, 0, 1, 0, 1, 0, 0] ],  # C01
        [ [0, 1, 0, 1, 0, 0, 0],   # C10
          [1, -1, 1, 0, 0, 1, 0] ]  # C11
    ]
    
    for k in range(7):
        for j in range(7):
            col = k * 7 + j
            
            # Compute U and V columns
            # U_col = sum_{idx} strassen_X[j][idx] * X_inputs[k][idx]
            u_vec = [0] * 16
            v_vec = [0] * 16
            
            for idx in range(4):
                coeff = strassen_X[j][idx]
                if coeff != 0:
                    for r in range(16):
                        u_vec[r] += coeff * X_inputs[k][idx][r]
                        
                coeff = strassen_Y[j][idx]
                if coeff != 0:
                    for r in range(16):
                        v_vec[r] += coeff * Y_inputs[k][idx][r]
                        
            for r in range(16):
                U[r][col] = u_vec[r]
                V[r][col] = v_vec[r]
                
            # Compute W column
            # W_row depends on C index.
            # C index w = 4*q + (2*p + r_local)
            # Coeff = alpha[p][q][k] * strassen_Z[r_local * 2 + s_local][j]
            
            for p in range(2):
                for q in range(2):
                    a_coeff = alpha[p][q][k]
                    if a_coeff == 0:
                        continue
                    for r_local in range(2):
                        for s_local in range(2):
                            z_coeff = strassen_Z[r_local * 2 + s_local][j]
                            if z_coeff == 0:
                                continue
                            val = a_coeff * z_coeff
                            w_idx = 4 * q + (2 * p + r_local)
                            W[w_idx][col] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
