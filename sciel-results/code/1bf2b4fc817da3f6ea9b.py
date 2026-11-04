import numpy as np
import time

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    rank = 97
    U = np.zeros((dim_u, rank), dtype=np.int8)
    V = np.zeros((dim_v, rank), dtype=np.int8)
    W = np.zeros((dim_w, rank), dtype=np.int8)
    
    col = 0
    
    # Part 2: Schoolbook for 4x4x3 (k in 4..6)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                a = i * 4 + j
                b = j * 7 + k
                c = k * 4 + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
                
    # Part 1: Strassen for 4x4x4 (k in 0..3)
    # Strassen products definitions for 4x4
    # s: 1..7
    # X_s: 2x2 matrix of coefficients for A blocks
    # Y_s: 2x2 matrix of coefficients for B blocks
    # C_blocks: list of (row_block, col_block) where result is added
    
    # A blocks: 00 (0,0), 01 (0,1), 10 (1,0), 11 (1,1)
    # B blocks: 00 (0,0), 01 (0,1), 10 (1,0), 11 (1,1)
    
    # Define X and Y for each s
    # X_s[u, v] is a linear form on A.
    # We represent the linear form by a 16-vector of coefficients.
    
    def get_A_vec(block_indices, signs):
        # block_indices: list of (r, c) for A blocks
        # signs: list of coefficients
        vec = np.zeros(16, dtype=np.int8)
        for (br, bc), sign in zip(block_indices, signs):
            # A block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
            for i in range(2):
                for j in range(2):
                    r = 2*br + i
                    c = 2*bc + j
                    a_idx = r * 4 + c
                    vec[a_idx] = sign
        return vec

    def get_B_vec(block_indices, signs):
        vec = np.zeros(28, dtype=np.int8)
        for (br, bc), sign in zip(block_indices, signs):
            # B block (br, bc) covers rows 2*br..2*br+1, cols 2*bc..2*bc+1
            # B is 4x7. But here we only use 4x4 part (cols 0..3).
            # Wait, B is 4x7. Blocks are 2x2.
            # B_{00} is rows 0,1 cols 0,1.
            # B_{01} is rows 0,1 cols 2,3.
            # B_{10} is rows 2,3 cols 0,1.
            # B_{11} is rows 2,3 cols 2,3.
            for i in range(2):
                for j in range(2):
                    r = 2*br + i
                    c = 2*bc + j
                    b_idx = r * 7 + c
                    vec[b_idx] = sign
        return vec

    # Strassen Level 1 (4x4)
    # s, X_blocks, X_signs, Y_blocks, Y_signs, C_target_blocks
    # C_target_blocks: list of (row_block, col_block, sign)
    
    strassen_defs = [
        # s=1: (A00+A11)(B00+B11) -> C00+1, C11+1
        (
            [(0,0), (1,1)], [1, 1],
            [(0,0), (1,1)], [1, 1],
            [(0,0, 1), (1,1, 1)]
        ),
        # s=2: (A10+A11)B00 -> C10+1, C11-1
        (
            [(1,0), (1,1)], [1, 1],
            [(0,0)], [1],
            [(1,0, 1), (1,1, -1)]
        ),
        # s=3: A00(B01-B11) -> C01+1, C11+1
        (
            [(0,0)], [1],
            [(0,1), (1,1)], [1, -1],
            [(0,1, 1), (1,1, 1)]
        ),
        # s=4: A11(B10-B00) -> C00+1, C10+1
        (
            [(1,1)], [1],
            [(1,0), (0,0)], [1, -1],
            [(0,0, 1), (1,0, 1)]
        ),
        # s=5: (A00+A01)B11 -> C00-1, C01+1
        (
            [(0,0), (0,1)], [1, 1],
            [(1,1)], [1],
            [(0,0, -1), (0,1, 1)]
        ),
        # s=6: (A10-A00)(B00+B01) -> C11+1
        (
            [(1,0), (0,0)], [1, -1],
            [(0,0), (0,1)], [1, 1],
            [(1,1, 1)]
        ),
        # s=7: (A01-A11)(B10+B11) -> C00+1
        (
            [(0,1), (1,1)], [1, -1],
            [(1,0), (1,1)], [1, 1],
            [(0,0, 1)]
        ),
    ]
    
    # Strassen Level 2 (2x2)
    # t, x_coeffs (for P), y_coeffs (for Q), w_coeffs (for C)
    # P, Q are 2x2 matrices.
    # x_coeffs: [p00, p01, p10, p11]
    # y_coeffs: [q00, q01, q10, q11]
    # w_coeffs: [c00, c01, c10, c11]
    
    strassen_2x2_defs = [
        # t=1: (p00+p11)(q00+q11) -> c00+1, c11+1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # t=2: (p10+p11)q00 -> c10+1, c11-1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # t=3: p00(q01-q11) -> c01+1, c11+1
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # t=4: p11(q10-q00) -> c00+1, c10+1
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # t=5: (p00+p01)q11 -> c00-1, c01+1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # t=6: (p10-p00)(q00+q01) -> c11+1
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # t=7: (p01-p11)(q10+q11) -> c00+1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    for s_idx, (X_blocks, X_signs, Y_blocks, Y_signs, C_targets) in enumerate(strassen_defs):
        # Get base vectors for X and Y (16 and 28 dims)
        # But X and Y are 2x2 matrices of linear forms.
        # We need to combine them with the 2x2 Strassen coefficients.
        
        # Pre-compute the 4 linear forms for X (for x00, x01, x10, x11)
        X_forms = []
        for u in range(2):
            for v in range(2):
                # X_s[u, v] is sum of A blocks with signs
                # Wait, X_s is defined as a matrix of blocks.
                # X_s[u, v] corresponds to the element at row u, col v of the 2x2 matrix X_s.
                # My definition of X_blocks above was for the whole matrix?
                # No, X_blocks was a list of blocks to sum.
                # This implies X_s is a single 2x2 matrix formed by summing blocks.
                # But Strassen requires X_s to be a specific 2x2 matrix.
                # E.g. X_1 = A_00 + A_11.
                # A_00 is 2x2. A_11 is 2x2.
                # X_1 is 2x2.
                # X_1[0,0] = A_00[0,0] + A_11[0,0].
                # So for each element (u,v) of X_s, we have a linear form on A.
                
                # Let's re-construct X_forms properly.
                # X_blocks is a list of (br, bc) and signs.
                # This defines the 2x2 matrix X_s.
                # X_s[u, v] = sum_{(br,bc), sign} sign * A_{br,bc}[u,v].
                
                vec = np.zeros(16, dtype=np.int8)
                for (br, bc), sign in zip(X_blocks, X_signs):
                    # A block (br, bc) element (u, v) is at global (2*br+u, 2*bc+v)
                    r = 2*br + u
                    c = 2*bc + v
                    a_idx = r * 4 + c
                    vec[a_idx] = sign
                X_forms.append(vec)
                
        Y_forms = []
        for u in range(2):
            for v in range(2):
                vec = np.zeros(28, dtype=np.int8)
                for (br, bc), sign in zip(Y_blocks, Y_signs):
                    r = 2*br + u
                    c = 2*bc + v
                    b_idx = r * 7 + c
                    vec[b_idx] = sign
                Y_forms.append(vec)
                
        for t_idx, (x_coeffs, y_coeffs, w_coeffs) in enumerate(strassen_2x2_defs):
            # Construct U, V, W for this term
            u_vec = np.zeros(16, dtype=np.int8)
            v_vec = np.zeros(28, dtype=np.int8)
            w_vec = np.zeros(28, dtype=np.int8)
            
            for idx in range(4):
                coeff = x_coeffs[idx]
                if coeff != 0:
                    u_vec += coeff * X_forms[idx]
                    
                coeff = y_coeffs[idx]
                if coeff != 0:
                    v_vec += coeff * Y_forms[idx]
            
            # Construct W
            # w_coeffs are for C_s[0,0], C_s[0,1], C_s[1,0], C_s[1,1]
            # Map to global C indices based on C_targets
            # C_targets is a list of (row_block, col_block, sign)
            # For each target, add the contribution.
            
            for (rb, cb, sign) in C_targets:
                for l in range(4):
                    coeff = w_coeffs[l]
                    if coeff != 0:
                        # Local index l corresponds to (l//2, l%2) in C_s?
                        # l=0: (0,0), l=1: (0,1), l=2: (1,0), l=3: (1,1)
                        lu = l // 2
                        lv = l % 2
                        # Global index in C
                        # C is 4x4.
                        # Block (rb, cb) covers rows 2*rb..2*rb+1, cols 2*cb..2*cb+1.
                        # Element (lu, lv) in block is at global (2*rb+lu, 2*cb+lv).
                        # c_idx = k * 4 + i.
                        # k is col, i is row.
                        k = 2*cb + lv
                        i = 2*rb + lu
                        c_idx = k * 4 + i
                        w_vec[c_idx] += sign * coeff
                        
            U[:, col] = u_vec
            V[:, col] = v_vec
            W[:, col] = w_vec
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
