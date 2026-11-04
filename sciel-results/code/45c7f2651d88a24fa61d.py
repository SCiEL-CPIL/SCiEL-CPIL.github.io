import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 decomposition
    rank = 49
    
    # U, V, W are 16 x 49
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define block indices for 4x4 matrices
    # Blocks: (0,0), (0,1), (1,0), (1,1)
    # Each block is 2x2.
    # A indices:
    # Block 0: (0,0), (0,1), (1,0), (1,1)
    # Block 1: (0,2), (0,3), (1,2), (1,3)
    # Block 2: (2,0), (2,1), (3,0), (3,1)
    # Block 3: (2,2), (2,3), (3,2), (3,3)
    
    # We need to define the linear forms for X_alpha and Y_alpha in terms of the scalar entries of A and B.
    # Let A_flat[i*4+j] = A[i,j]
    # Let B_flat[i*4+j] = B[i,j]
    # Let C_flat[i*4+j] = C[i,j]
    
    # Helper to get flat index
    def idx(i, j):
        return i * 4 + j
    
    # Outer Strassen definitions
    # alpha: 0 to 6
    # X_alpha coefficients on A blocks
    # Y_alpha coefficients on B blocks
    
    # Block linear forms for X_alpha (2x2 matrix of linear forms on A)
    # X_alpha = sum_{b in 0..3} coeff_alpha_b * A_block_b
    # A_block_b is a 2x2 matrix.
    # So X_alpha[i,j] = sum_b coeff_alpha_b * A_block_b[i,j]
    
    # Let's define coeff_alpha_b for A (X) and B (Y)
    # alpha 0: X = A0 + A3, Y = B0 + B3
    # alpha 1: X = A2 + A3, Y = B0
    # alpha 2: X = A0, Y = B3 - B1
    # alpha 3: X = A3, Y = B2 - B0
    # alpha 4: X = A0 + A1, Y = B3
    # alpha 5: X = A2 - A0, Y = B0 + B1
    # alpha 6: X = A1 - A3, Y = B2 + B3
    
    coeff_A = [
        [(1, 0), (0, 0), (0, 0), (1, 0)], # 0
        [(0, 0), (0, 0), (1, 0), (1, 0)], # 1
        [(1, 0), (0, 0), (0, 0), (0, 0)], # 2
        [(0, 0), (0, 0), (0, 0), (1, 0)], # 3
        [(1, 0), (1, 0), (0, 0), (0, 0)], # 4
        [(-1, 0), (0, 0), (1, 0), (0, 0)], # 5
        [(0, 0), (1, 0), (0, 0), (-1, 0)] # 6
    ]
    
    coeff_B = [
        [(1, 0), (0, 0), (0, 0), (1, 0)], # 0
        [(1, 0), (0, 0), (0, 0), (0, 0)], # 1
        [(0, 0), (-1, 0), (0, 0), (1, 0)], # 2
        [(0, 0), (0, 0), (1, 0), (-1, 0)], # 3
        [(0, 0), (0, 0), (0, 0), (1, 0)], # 4
        [(1, 0), (1, 0), (0, 0), (0, 0)], # 5
        [(0, 0), (0, 0), (1, 0), (1, 0)] # 6
    ]
    
    # For each alpha, beta
    for alpha in range(7):
        # Construct X_alpha and Y_alpha as 2x2 matrices of coefficients on A_flat and B_flat
        # X_alpha[i][j] is a list of (flat_index, coeff) for A
        # Y_alpha[i][j] is a list of (flat_index, coeff) for B
        
        X = [[[] for _ in range(2)] for _ in range(2)]
        Y = [[[] for _ in range(2)] for _ in range(2)]
        
        for b in range(4):
            ca = coeff_A[alpha][b][0]
            cb = coeff_B[alpha][b][0]
            
            if ca != 0:
                # Block b is at rows 2*(b//2), 2*(b//2)+1 and cols 2*(b%2), 2*(b%2)+1
                r0, c0 = 2 * (b // 2), 2 * (b % 2)
                for dr in range(2):
                    for dc in range(2):
                        r, c = r0 + dr, c0 + dc
                        flat_idx = idx(r, c)
                        X[dr][dc].append((flat_idx, ca))
                        
            if cb != 0:
                r0, c0 = 2 * (b // 2), 2 * (b % 2)
                for dr in range(2):
                    for dc in range(2):
                        r, c = r0 + dr, c0 + dc
                        flat_idx = idx(r, c)
                        Y[dr][dc].append((flat_idx, cb))
        
        # Inner Strassen for X * Y
        # X = [[x00, x01], [x10, x11]]
        # Y = [[y00, y01], [y10, y11]]
        
        # Define inner Strassen linear forms
        # u0 = x00 + x11
        # u1 = x10 + x11
        # u2 = x00
        # u3 = x11
        # u4 = x00 + x01
        # u5 = x10 - x00
        # u6 = x01 - x11
        
        # v0 = y00 + y11
        # v1 = y00
        # v2 = y11 - y01
        # v3 = y10 - y00
        # v4 = y11
        # v5 = y00 + y01
        # v6 = y10 + y11
        
        # We need to combine the linear forms.
        # Since each x_ij and y_ij is a sum of (flat_idx, coeff), the sum is also a sum.
        
        def combine(forms1, forms2, sign1=1, sign2=1):
            # forms1, forms2 are lists of (idx, coeff)
            # Returns a list of (idx, coeff) for sign1*f1 + sign2*f2
            # Actually, let's just create a dict {idx: coeff}
            res = {}
            for idx, c in forms1:
                res[idx] = res.get(idx, 0) + c * sign1
            for idx, c in forms2:
                res[idx] = res.get(idx, 0) + c * sign2
            return res
        
        # Construct u_beta and v_beta for beta in 0..6
        u_forms = [
            combine(X[0][0], X[1][1]), # u0
            combine(X[1][0], X[1][1]), # u1
            X[0][0][:], # u2
            X[1][1][:], # u3
            combine(X[0][0], X[0][1]), # u4
            combine(X[1][0], X[0][0], 1, -1), # u5
            combine(X[0][1], X[1][1], 1, -1) # u6
        ]
        
        v_forms = [
            combine(Y[0][0], Y[1][1]), # v0
            Y[0][0][:], # v1
            combine(Y[1][1], Y[0][1], 1, -1), # v2
            combine(Y[1][0], Y[0][0], 1, -1), # v3
            Y[1][1][:], # v4
            combine(Y[0][0], Y[0][1]), # v5
            combine(Y[1][0], Y[1][1]) # v6
        ]
        
        # Inner Strassen output coefficients for M_alpha entries
        # M00 = m0 + m3 - m4 + m6
        # M01 = m2 + m4
        # M10 = m1 + m3
        # M11 = m0 - m1 + m2 + m5
        
        # For each beta (0..6), we determine which M entries it contributes to
        # and with what coefficient.
        inner_coeffs = [
            (1, 0, 0, 1), # beta 0 -> M00, M11
            (0, 0, 1, -1), # beta 1 -> M10, M11
            (0, 1, 0, 1), # beta 2 -> M01, M11
            (1, 0, 1, 0), # beta 3 -> M00, M10
            (-1, 1, 0, 0), # beta 4 -> M00, M01
            (0, 0, 0, 1), # beta 5 -> M11
            (1, 0, 0, 0) # beta 6 -> M00
        ]
        
        # Outer Strassen distribution of M_alpha to C blocks
        # C00 gets M0, M3, -M4, M6
        # C01 gets M2, M4
        # C10 gets M1, M3
        # C11 gets M0, -M1, M2, M5
        
        # Block (br, bc) of C corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1
        # M_alpha is a block. Its entries M_alpha[ur, uc] map to C[2*br+ur, 2*bc+uc]
        
        # For each alpha, define the set of (block_row, block_col, sign)
        outer_dist = [
            [(0, 0, 1), (1, 1, 1)], # alpha 0
            [(1, 0, 1), (1, 1, -1)], # alpha 1
            [(0, 1, 1), (1, 1, 1)], # alpha 2
            [(0, 0, 1), (1, 0, 1)], # alpha 3
            [(0, 0, -1), (0, 1, 1)], # alpha 4
            [(1, 1, 1)], # alpha 5
            [(0, 0, 1)] # alpha 6
        ]
        
        for beta in range(7):
            t = alpha * 7 + beta
            
            u_dict = u_forms[beta]
            v_dict = v_forms[beta]
            
            # Fill U and V
            for idx_a, coeff_a in u_dict.items():
                U[idx_a][t] = coeff_a
            for idx_b, coeff_b in v_dict.items():
                V[idx_b][t] = coeff_b
            
            # Fill W
            # m_{alpha, beta} contributes to M_alpha entries with coeffs from inner_coeffs[beta]
            m00_c, m01_c, m10_c, m11_c = inner_coeffs[beta]
            
            # For each destination block in C defined by outer_dist[alpha]
            for br, bc, sign in outer_dist[alpha]:
                # M_alpha[0,0] -> C[2*br, 2*bc]
                if m00_c != 0:
                    idx_c = idx(2*br, 2*bc)
                    W[idx_c][t] += m00_c * sign
                # M_alpha[0,1] -> C[2*br, 2*bc+1]
                if m01_c != 0:
                    idx_c = idx(2*br, 2*bc+1)
                    W[idx_c][t] += m01_c * sign
                # M_alpha[1,0] -> C[2*br+1, 2*bc]
                if m10_c != 0:
                    idx_c = idx(2*br+1, 2*bc)
                    W[idx_c][t] += m10_c * sign
                # M_alpha[1,1] -> C[2*br+1, 2*bc+1]
                if m11_c != 0:
                    idx_c = idx(2*br+1, 2*bc+1)
                    W[idx_c][t] += m11_c * sign
            
    return {"U": U, "V": V, "W": W}
