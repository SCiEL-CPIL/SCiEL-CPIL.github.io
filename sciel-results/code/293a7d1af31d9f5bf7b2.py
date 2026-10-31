import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Base 2x2 Strassen coefficients
    # u_q, v_q are vectors of length 4 (flattened 2x2)
    # w_q is vector of length 4 (flattened 2x2 output)
    u_base = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 1], # 2
        [1, 0, 0, 0], # 3
        [0, 0, 0, 1], # 4
        [1, 1, 0, 0], # 5
        [-1, 0, 1, 0],# 6
        [0, 1, 0, -1] # 7
    ]
    v_base = [
        [1, 0, 0, 1], # 1
        [1, 0, 0, 0], # 2
        [0, 1, 0, -1],# 3
        [0, -1, 1, 0],# 4
        [0, 0, 0, 1], # 5
        [1, 1, 0, 0], # 6
        [0, 0, 1, 1]  # 7
    ]
    w_base = [
        [1, 0, 0, 1], # 1
        [0, 0, 1, 0], # 2
        [0, 1, 0, 0], # 3
        [1, 0, 1, 0], # 4
        [-1, 1, 0, 0],# 5
        [0, 0, 0, 1], # 6
        [1, 0, 0, 1]  # 7
    ]

    # Block mappings for X(l) and Y(l)
    # A blocks: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # X(l) is 2x2 matrix of linear forms on A
    # X(l)_{ij} is a linear form on the 16 entries of A
    # We represent X(l)_{ij} as a vector of 16 coefficients
    
    def get_block_coeffs(block_idx, sign=1):
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11
        # Returns vector of 16 coeffs
        coeffs = [0] * 16
        if block_idx == 0:
            indices = [0, 1, 4, 5]
        elif block_idx == 1:
            indices = [2, 3, 6, 7]
        elif block_idx == 2:
            indices = [8, 9, 12, 13]
        else:
            indices = [10, 11, 14, 15]
        for idx in indices:
            coeffs[idx] = sign
        return coeffs

    def add_vecs(v1, v2, s1=1, s2=1):
        return [s1*a + s2*b for a, b in zip(v1, v2)]

    # Define X(l) and Y(l) as 4 vectors of 16 coeffs each (for the 4 entries of the 2x2 matrix)
    # X(l) = [[X00, X01], [X10, X11]]
    X_defs = []
    Y_defs = []
    
    # l=1: X = A00+A11, Y = B00+B11
    x1 = [add_vecs(get_block_coeffs(0), get_block_coeffs(3)), [0]*16, [0]*16, add_vecs(get_block_coeffs(0), get_block_coeffs(3))]
    # Wait, X(l) is a 2x2 matrix.
    # For Strassen, X(l) is a matrix.
    # X(1) = A00 + A11. This is a 2x2 matrix.
    # So X(1)_{00} = (A00+A11)_{00}, etc.
    # My get_block_coeffs returns coeffs for the whole 2x2 block.
    # So X(1) is just one 2x2 matrix.
    # But Strassen formulas like X(2) = A10 + A11 also result in a 2x2 matrix.
    # So for all l, X(l) is a 2x2 matrix.
    # The entries of X(l) are linear forms.
    # So X(l) can be represented by 4 vectors of length 16.
    
    # Let's redefine:
    # X(l) is a 2x2 matrix.
    # X(l)_{00}, X(l)_{01}, X(l)_{10}, X(l)_{11} are linear forms.
    
    # l=1: X = A00+A11.
    # X_{00} = (A00+A11)_{00} = A00_{00} + A11_{00}
    # This is getting complicated to map indices.
    # Easier: X(l) is a 2x2 matrix of blocks? No.
    # A is 4x4. A00 is 2x2.
    # X(1) = A00 + A11 is a 2x2 matrix.
    # So X(1) has 4 entries.
    # Entry (0,0) of X(1) is Entry (0,0) of A00 + Entry (0,0) of A11.
    
    # Let's build the 16-length vectors for each of the 4 entries of X(l).
    
    def make_x(l):
        # Returns list of 4 vectors of length 16
        res = [[0]*16 for _ in range(4)]
        if l == 1:
            # X = A00 + A11
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a00 = i*4+j
                    idx_a11 = 10 + i*4+j
                    res[idx_x][idx_a00] = 1
                    res[idx_x][idx_a11] = 1
        elif l == 2:
            # X = A10 + A11
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a10 = 8 + i*4+j
                    idx_a11 = 10 + i*4+j
                    res[idx_x][idx_a10] = 1
                    res[idx_x][idx_a11] = 1
        elif l == 3:
            # X = A00
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a00 = i*4+j
                    res[idx_x][idx_a00] = 1
        elif l == 4:
            # X = A11
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a11 = 10 + i*4+j
                    res[idx_x][idx_a11] = 1
        elif l == 5:
            # X = A00 + A01
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a00 = i*4+j
                    idx_a01 = 2 + i*4+j
                    res[idx_x][idx_a00] = 1
                    res[idx_x][idx_a01] = 1
        elif l == 6:
            # X = A10 - A00
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a10 = 8 + i*4+j
                    idx_a00 = i*4+j
                    res[idx_x][idx_a10] = 1
                    res[idx_x][idx_a00] = -1
        elif l == 7:
            # X = A01 - A11
            for i in range(2):
                for j in range(2):
                    idx_x = i*2+j
                    idx_a01 = 2 + i*4+j
                    idx_a11 = 10 + i*4+j
                    res[idx_x][idx_a01] = 1
                    res[idx_x][idx_a11] = -1
        return res

    def make_y(l):
        res = [[0]*16 for _ in range(4)]
        if l == 1:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b00 = i*4+j
                    idx_b11 = 10 + i*4+j
                    res[idx_y][idx_b00] = 1
                    res[idx_y][idx_b11] = 1
        elif l == 2:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b00 = i*4+j
                    res[idx_y][idx_b00] = 1
        elif l == 3:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b01 = 2 + i*4+j
                    idx_b11 = 10 + i*4+j
                    res[idx_y][idx_b01] = 1
                    res[idx_y][idx_b11] = -1
        elif l == 4:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b10 = 8 + i*4+j
                    idx_b00 = i*4+j
                    res[idx_y][idx_b10] = 1
                    res[idx_y][idx_b00] = -1
        elif l == 5:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b11 = 10 + i*4+j
                    res[idx_y][idx_b11] = 1
        elif l == 6:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b00 = i*4+j
                    idx_b01 = 2 + i*4+j
                    res[idx_y][idx_b00] = 1
                    res[idx_y][idx_b01] = 1
        elif l == 7:
            for i in range(2):
                for j in range(2):
                    idx_y = i*2+j
                    idx_b10 = 8 + i*4+j
                    idx_b11 = 10 + i*4+j
                    res[idx_y][idx_b10] = 1
                    res[idx_y][idx_b11] = 1
        return res

    X_defs = [make_x(l) for l in range(1, 8)]
    Y_defs = [make_y(l) for l in range(1, 8)]

    # Gamma coefficients for C_ij = sum_l gamma_l C_l
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    gamma = [
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],  # C01
        [0, 1, 0, 1, 0, 0, 0],  # C10
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ]

    for l_idx in range(7):
        l = l_idx + 1
        X_l = X_defs[l_idx]
        Y_l = Y_defs[l_idx]
        
        for q_idx in range(7):
            r = l_idx * 7 + q_idx
            u_q = u_base[q_idx]
            v_q = v_base[q_idx]
            w_q = w_base[q_idx]
            
            # Construct U vector (length 16)
            U_vec = [0] * 16
            for k in range(4): # k is index in 2x2 X matrix (0..3)
                coeff = u_q[k]
                if coeff != 0:
                    for idx in range(16):
                        U_vec[idx] += coeff * X_l[k][idx]
                        
            # Construct V vector (length 16)
            V_vec = [0] * 16
            for k in range(4):
                coeff = v_q[k]
                if coeff != 0:
                    for idx in range(16):
                        V_vec[idx] += coeff * Y_l[k][idx]
                        
            # Construct W vector (length 16)
            W_vec = [0] * 16
            for block_idx in range(4): # 0=C00, 1=C01, 2=C10, 3=C11
                coeff_block = gamma[block_idx][l_idx]
                coeff_w = w_q[block_idx]
                val = coeff_block * coeff_w
                if val != 0:
                    # Fill the 2x2 block
                    if block_idx == 0:
                        indices = [0, 1, 4, 5]
                    elif block_idx == 1:
                        indices = [2, 3, 6, 7]
                    elif block_idx == 2:
                        indices = [8, 9, 12, 13]
                    else:
                        indices = [10, 11, 14, 15]
                    for idx in indices:
                        W_vec[idx] = val

            # Assign to U, V, W
            for i in range(16):
                U[i][r] = U_vec[i]
                V[i][r] = V_vec[i]
                W[i][r] = W_vec[i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
