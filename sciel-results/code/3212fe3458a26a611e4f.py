import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen term for 2x2x2 to the global matrices
    # local_u: 4x1 vector (flattened 2x2)
    # local_v: 4x1 vector
    # local_w: 4x1 vector
    # block_i, block_j: block indices for A and B (0..1)
    # block_k: block index for C (0..1) -- wait, C blocks are determined by W
    
    # We will generate the 49 terms for the first 4x4x4 part (C1)
    # and 49 terms for the second 4x4x4 part (C2)
    
    # Strassen terms for 2x2x2
    # Returns list of (u_vec, v_vec, w_vec)
    # u_vec, v_vec, w_vec are length 4 lists (flattened 2x2)
    def get_strassen_2x2_terms():
        terms = []
        # M1: (a+d)(e+h) -> C00(1), C11(1)
        u = [1, 0, 0, 1]
        v = [1, 0, 0, 1]
        w = [1, 0, 0, 1]
        terms.append((u, v, w))
        
        # M2: (c+d)e -> C10(1), C11(1)
        u = [0, 0, 1, 1]
        v = [1, 0, 0, 0]
        w = [0, 0, 1, 1]
        terms.append((u, v, w))
        
        # M3: a(f-h) -> C01(1), C11(1)
        u = [1, 0, 0, 0]
        v = [0, 1, 0, -1]
        w = [0, 1, 0, 1]
        terms.append((u, v, w))
        
        # M4: d(g-e) -> C00(1), C10(1)
        u = [0, 0, 0, 1]
        v = [0, 0, 1, -1] # g-e -> 0,0,1,-1 (e is 0,0; g is 1,0)
        w = [1, 0, 1, 0]
        terms.append((u, v, w))
        
        # M5: (a+b)h -> C00(-1), C01(1)
        u = [1, 1, 0, 0]
        v = [0, 0, 0, 1]
        w = [-1, 1, 0, 0]
        terms.append((u, v, w))
        
        # M6: (c-a)(e+f) -> C11(1)
        u = [0, 0, 1, -1]
        v = [1, 1, 0, 0]
        w = [0, 0, 0, 1]
        terms.append((u, v, w))
        
        # M7: (b-d)(g+h) -> C00(1), C11(-1)
        u = [0, 1, 0, -1]
        v = [0, 0, 1, 1]
        w = [1, 0, 0, -1]
        terms.append((u, v, w))
        
        return terms

    strassen_terms = get_strassen_2x2_terms()
    
    # Define the 7 block products for 4x4x4
    # Each is a tuple of (u_blocks, v_blocks, w_blocks)
    # u_blocks: dict from block_id (0..3) to coefficient
    # v_blocks: dict from block_id (0..3) to coefficient
    # w_blocks: dict from block_id (0..3) to coefficient
    # Block ids: 0=A00, 1=A01, 2=A10, 3=A11
    
    block_products = []
    
    # P1: (A00+A11)(B00+B11)
    u_blocks = {0: 1, 3: 1}
    v_blocks = {0: 1, 3: 1}
    w_blocks = {0: 1, 3: 1} # Result is added to C00 and C11? No, P1 is a 2x2 matrix.
    # The W part of the decomposition of P1 determines how P1's 2x2 result maps to C's 4x2 blocks.
    # P1 contributes to C00 and C11.
    # Specifically, C00 += P1[0,0] + P1[1,1] ... wait.
    # In Strassen 4x4, the block products are just intermediate 2x2 matrices.
    # The final C blocks are linear combinations of these P_k.
    # We are decomposing the tensor for the whole 4x4x4 multiplication.
    # The Strassen algorithm computes C = f(P1..P7).
    # We can treat each P_k = X_k Y_k as a 2x2x2 multiplication.
    # So we decompose P_k = sum_{t=1}^7 u_{kt} v_{kt} w_{kt}.
    # Then C = sum_k f_k(P_k) = sum_k sum_t f_k(u_{kt} v_{kt} w_{kt}).
    # This is not a direct sum of rank 1 terms for C.
    # We need to push the linear combinations into the U, V, W factors.
    
    # Let's define the 7 linear combinations for C blocks in terms of P_k.
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # So, for each term t in the decomposition of P_k, we have a rank-1 term for P_k.
    # This term contributes to C blocks according to the coefficients in the above equations.
    
    # So, for each k in 1..7 and t in 1..7:
    #   u_global = expand(u_blocks_k, u_strassen_t)
    #   v_global = expand(v_blocks_k, v_strassen_t)
    #   w_global = expand(w_combinations, w_strassen_t)
    
    # w_combinations is a dict mapping C_block_id to coefficient for P_k.
    
    c_combinations = [
        {0: 1, 3: 1},   # P1 -> C00(1), C11(1) -- Wait, P1 is a 2x2 matrix.
        # P1 is not a scalar. P1 is a 2x2 matrix.
        # The formula C00 = P1[0,0] + P1[1,1] ... is NOT correct.
        # Strassen's algorithm for 4x4:
        # M1 = (A00+A11)(B00+B11) -> 2x2 matrix
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So M_k are 2x2 matrices.
        # C00 (2x2) = M1 (2x2) + M4 (2x2) - M5 (2x2) + M7 (2x2).
        # So the coefficient of M1 in C00 is 1 (identity matrix).
        # The coefficient of M4 in C00 is 1.
        # The coefficient of M5 in C00 is -1.
        # The coefficient of M7 in C00 is 1.
    ]
    
    # Correct combinations (coefficient of M_k in each C_block):
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    coeffs = [
        {0: 1, 3: 1}, # M1: C00(1), C11(1)
        {2: 1, 3: -1}, # M2: C10(1), C11(-1)
        {1: 1, 3: 1}, # M3: C01(1), C11(1)
        {0: 1, 2: 1}, # M4: C00(1), C10(1)
        {0: -1, 1: 1}, # M5: C00(-1), C01(1)
        {3: 1}, # M6: C11(1)
        {0: 1, 3: -1} # M7: C00(1), C11(-1)
    ]
    
    u_block_defs = [
        {0: 1, 3: 1}, # M1
        {2: 1, 3: 1}, # M2
        {0: 1}, # M3
        {3: 1}, # M4
        {0: 1, 1: 1}, # M5
        {2: 1, 0: -1}, # M6
        {1: 1, 3: -1} # M7
    ]
    
    v_block_defs = [
        {0: 1, 3: 1}, # M1
        {0: 1}, # M2
        {1: 1, 3: -1}, # M3
        {2: 1, 0: -1}, # M4
        {3: 1}, # M5
        {0: 1, 1: 1}, # M6
        {2: 1, 3: 1} # M7
    ]
    
    col = 0
    for k in range(7):
        # For each of the 7 Strassen terms within M_k
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            # Expand u_loc to global U
            for bi in range(4): # A block id
                if u_block_defs[k].get(bi, 0) != 0:
                    coeff_u = u_block_defs[k][bi]
                    # u_loc is flattened 2x2: [00, 01, 10, 11]
                    for r in range(2):
                        for c in range(2):
                            val = u_loc[r*2+c]
                            if val != 0:
                                i_global = bi//2 * 2 + r
                                j_global = bi%2 * 2 + c
                                u_idx = i_global * m + j_global
                                U[u_idx][col] += coeff_u * val
            
            # Expand v_loc to global V
            for bj in range(4): # B block id
                if v_block_defs[k].get(bj, 0) != 0:
                    coeff_v = v_block_defs[k][bj]
                    for r in range(2):
                        for c in range(2):
                            val = v_loc[r*2+c]
                            if val != 0:
                                j_global = bj//2 * 2 + r
                                k_local = bj%2 * 2 + c
                                # This is for C1 (k=0..3)
                                v_idx = j_global * p + k_local
                                V[v_idx][col] += coeff_v * val
            
            # Expand w_loc to global W
            # w_loc is flattened 2x2 result of M_k.
            # coeffs[k] maps M_k to C blocks.
            for bc in range(4): # C block id
                if coeffs[k].get(bc, 0) != 0:
                    coeff_w = coeffs[k][bc]
                    for r in range(2):
                        for c in range(2):
                            val = w_loc[r*2+c]
                            if val != 0:
                                k_global = bc//2 * 2 + r
                                i_global = bc%2 * 2 + c
                                w_idx = k_global * n + i_global
                                W[w_idx][col] += coeff_w * val
            
            col += 1
    
    # Now add the second 4x4x4 part (C2, k=4..7)
    # Same U, V, W structure, but V and W indices shifted.
    for k in range(7):
        for t in range(7):
            u_loc, v_loc, w_loc = strassen_terms[t]
            
            for bi in range(4):
                if u_block_defs[k].get(bi, 0) != 0:
                    coeff_u = u_block_defs[k][bi]
                    for r in range(2):
                        for c in range(2):
                            val = u_loc[r*2+c]
                            if val != 0:
                                i_global = bi//2 * 2 + r
                                j_global = bi%2 * 2 + c
                                u_idx = i_global * m + j_global
                                U[u_idx][col] += coeff_u * val
            
            for bj in range(4):
                if v_block_defs[k].get(bj, 0) != 0:
                    coeff_v = v_block_defs[k][bj]
                    for r in range(2):
                        for c in range(2):
                            val = v_loc[r*2+c]
                            if val != 0:
                                j_global = bj//2 * 2 + r
                                k_local = bj%2 * 2 + c
                                v_idx = j_global * p + (4 + k_local)
                                V[v_idx][col] += coeff_v * val
            
            for bc in range(4):
                if coeffs[k].get(bc, 0) != 0:
                    coeff_w = coeffs[k][bc]
                    for r in range(2):
                        for c in range(2):
                            val = w_loc[r*2+c]
                            if val != 0:
                                k_global = bc//2 * 2 + r
                                i_global = bc%2 * 2 + c
                                w_idx = (4 + k_global) * n + i_global
                                W[w_idx][col] += coeff_w * val
            
            col += 1
    
    # Now add the 4x4x1 part (C3, k=8)
    # Schoolbook
    for i in range(4):
        for j in range(4):
            u_idx = i * m + j
            v_idx = j * p + 8
            w_idx = 8 * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

