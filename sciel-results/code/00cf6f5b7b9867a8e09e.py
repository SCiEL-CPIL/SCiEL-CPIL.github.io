import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    
    def time_remaining():
        return deadline - time.time()

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        # Check shapes
        if U_arr.shape[0] != n * m: return False
        if V_arr.shape[0] != m * p: return False
        if W_arr.shape[0] != p * n: return False
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # T_ijk = sum_r U_ijn_r * V_jkn_r * W_kin_r
        # We want T_ijk = 1 for all i, j, k
        T = np.einsum('ijn,jkn,kni->ijk', U_3d, V_3d, W_3d)
        
        # The tensor for matrix multiplication is 1 everywhere.
        # Wait, is it?
        # C_ik = sum_j A_ij B_jk.
        # So T_ijk = 1 if we are decomposing the multiplication tensor.
        # Yes.
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True

    def get_schoolbook():
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank), dtype=float)
        V = np.zeros((m * p, schoolbook_rank), dtype=float)
        W = np.zeros((p * n, schoolbook_rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Generate rank 49 for 4x4x4
    def generate_49_terms():
        # We will store terms as lists of dicts
        # U_terms: list of dict {(i, j): coeff}
        # V_terms: list of dict {(j, k): coeff}
        # W_terms: list of dict {(k, i): coeff}
        
        U_terms = []
        V_terms = []
        W_terms = []
        
        # Helper to add a term
        def add_term(u_dict, v_dict, w_dict):
            U_terms.append(u_dict)
            V_terms.append(v_dict)
            W_terms.append(w_dict)
            
        # Helper to add to a dict
        def add_to_dict(d, key, val):
            if val == 0: return
            d[key] = d.get(key, 0) + val
            if d[key] == 0:
                del d[key]
                
        # Strassen 2x2 definitions
        # u_combinations: list of 7 dicts for U (2x2)
        # v_combinations: list of 7 dicts for V (2x2)
        # w_combinations: list of 4x4 coeffs for C (2x2)
        
        # M1 = (A00+A11)(B00+B11)
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
        
        # For the outer Strassen (4x4 as 2x2 blocks of 2x2):
        # Blocks of A: A00, A01, A10, A11
        # Blocks of B: B00, B01, B10, B11
        # Blocks of C: C00, C01, C10, C11
        
        # We define the outer U and V combinations as dicts of blocks.
        # Key: (block_i, block_j) -> dict of local entries?
        # No, let's keep it simple.
        # The outer Strassen generates 7 terms.
        # Each term is a product of two 2x2 matrices (blocks).
        # Let's call the block matrices U_block and V_block.
        # U_block and V_block are 2x2 matrices where each entry is a 2x2 matrix.
        # We will represent U_block as a dict: (bi, bj) -> dict of (li, lj) -> coeff
        # where bi, bj are block indices (0,1) and li, lj are local indices (0,1).
        # The global index for A is (2*bi + li, 2*bj + lj).
        
        outer_U = []
        outer_V = []
        outer_W = [] # This will store the combination of M's for C
        
        # Define M1..M7
        # M1
        u1 = {}
        v1 = {}
        # U1 = A00 + A11
        for bi, bj in [(0,0), (1,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(u1, (bi, bj, li, lj), 1)
        # V1 = B00 + B11
        for bi, bj in [(0,0), (1,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(v1, (bi, bj, li, lj), 1)
        outer_U.append(u1)
        outer_V.append(v1)
        
        # M2
        u2 = {}
        v2 = {}
        # U2 = A10 + A11
        for bi, bj in [(1,0), (1,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(u2, (bi, bj, li, lj), 1)
        # V2 = B00
        for li in range(2):
            for lj in range(2):
                add_to_dict(v2, (0, 0, li, lj), 1)
        outer_U.append(u2)
        outer_V.append(v2)
        
        # M3
        u3 = {}
        v3 = {}
        # U3 = A00
        for li in range(2):
            for lj in range(2):
                add_to_dict(u3, (0, 0, li, lj), 1)
        # V3 = B01 - B11
        for li in range(2):
            for lj in range(2):
                add_to_dict(v3, (0, 1, li, lj), 1)
                add_to_dict(v3, (1, 1, li, lj), -1)
        outer_U.append(u3)
        outer_V.append(v3)
        
        # M4
        u4 = {}
        v4 = {}
        # U4 = A11
        for li in range(2):
            for lj in range(2):
                add_to_dict(u4, (1, 1, li, lj), 1)
        # V4 = B10 - B00
        for li in range(2):
            for lj in range(2):
                add_to_dict(v4, (1, 0, li, lj), 1)
                add_to_dict(v4, (0, 0, li, lj), -1)
        outer_U.append(u4)
        outer_V.append(v4)
        
        # M5
        u5 = {}
        v5 = {}
        # U5 = A00 + A01
        for bi, bj in [(0,0), (0,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(u5, (bi, bj, li, lj), 1)
        # V5 = B11
        for li in range(2):
            for lj in range(2):
                add_to_dict(v5, (1, 1, li, lj), 1)
        outer_U.append(u5)
        outer_V.append(v5)
        
        # M6
        u6 = {}
        v6 = {}
        # U6 = A10 - A00
        for li in range(2):
            for lj in range(2):
                add_to_dict(u6, (1, 0, li, lj), 1)
                add_to_dict(u6, (0, 0, li, lj), -1)
        # V6 = B00 + B01
        for bi, bj in [(0,0), (0,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(v6, (bi, bj, li, lj), 1)
        outer_U.append(u6)
        outer_V.append(v6)
        
        # M7
        u7 = {}
        v7 = {}
        # U7 = A01 - A11
        for li in range(2):
            for lj in range(2):
                add_to_dict(u7, (0, 1, li, lj), 1)
                add_to_dict(u7, (1, 1, li, lj), -1)
        # V7 = B10 + B11
        for bi, bj in [(1,0), (1,1)]:
            for li in range(2):
                for lj in range(2):
                    add_to_dict(v7, (bi, bj, li, lj), 1)
        outer_U.append(u7)
        outer_V.append(v7)
        
        # Now, for each outer term k (0..6), we have u_k and v_k which are 2x2 block matrices.
        # We need to multiply them using inner Strassen.
        # The inner Strassen operates on 2x2 matrices.
        # The "entries" of u_k are 2x2 matrices.
        # Let's extract the 2x2 matrices for u_k and v_k.
        # u_k is a dict: (bi, bj, li, lj) -> coeff.
        # We can group by (bi, bj) to get the 2x2 block structure?
        # No, u_k is a 2x2 matrix of 2x2 blocks.
        # Let's represent u_k as a 2x2 numpy array of 2x2 numpy arrays?
        # Or just use the local Strassen logic directly on the flattened indices?
        
        # It's easier to apply the local Strassen definitions.
        # Local Strassen M1' = (U'00 + U'11)(V'00 + V'11)
        # Here U'00 is the block (0,0) of u_k, which is a 2x2 matrix.
        # Let's define a function to get the block (bi, bj) of u_k as a dict of local entries.
        
        def get_block(u_dict, bi, bj):
            block = {}
            for key, val in u_dict.items():
                kbi, kbj, kli, klj = key
                if kbi == bi and kbj == bj:
                    block[(kli, klj)] = val
            return block
            
        # Inner Strassen terms for a 2x2 matrix product U' * V'
        # U' and V' are 2x2 matrices of scalars (in our case, the entries are the global A/B coeffs).
        # Wait, the entries of U' are *themselves* linear combinations of global A.
        # So when we do U'00 + U'11, we are adding the dicts.
        
        def add_dicts(d1, d2):
            d_new = {}
            for k, v in d1.items():
                add_to_dict(d_new, k, v)
            for k, v in d2.items():
                add_to_dict(d_new, k, v)
            return d_new
            
        def sub_dicts(d1, d2):
            d_new = {}
            for k, v in d1.items():
                add_to_dict(d_new, k, v)
            for k, v in d2.items():
                add_to_dict(d_new, k, -v)
            return d_new
            
        for k in range(7):
            u_k = outer_U[k]
            v_k = outer_V[k]
            
            # Get the 4 blocks of u_k and v_k
            u00 = get_block(u_k, 0, 0)
            u01 = get_block(u_k, 0, 1)
            u10 = get_block(u_k, 1, 0)
            u11 = get_block(u_k, 1, 1)
            
            v00 = get_block(v_k, 0, 0)
            v01 = get_block(v_k, 0, 1)
            v10 = get_block(v_k, 1, 0)
            v11 = get_block(v_k, 1, 1)
            
            # Inner Strassen Multiplications
            # M1' = (u00 + u11)(v00 + v11)
            # M2' = (u10 + u11)v00
            # M3' = u00(v01 - v11)
            # M4' = u11(v10 - v00)
            # M5' = (u00 + u01)v11
            # M6' = (u10 - u00)(v00 + v01)
            # M7' = (u01 - u11)(v10 + v11)
            
            inner_u = []
            inner_v = []
            
            inner_u.append(add_dicts(u00, u11))
            inner_v.append(add_dicts(v00, v11))
            
            inner_u.append(add_dicts(u10, u11))
            inner_v.append(v00)
            
            inner_u.append(u00)
            inner_v.append(sub_dicts(v01, v11))
            
            inner_u.append(u11)
            inner_v.append(sub_dicts(v10, v00))
            
            inner_u.append(add_dicts(u00, u01))
            inner_v.append(v11)
            
            inner_u.append(sub_dicts(u10, u00))
            inner_v.append(add_dicts(v00, v01))
            
            inner_u.append(sub_dicts(u01, u11))
            inner_v.append(add_dicts(v10, v11))
            
            # Now we have 7 scalar multiplications for this outer term k.
            # We need to determine the W coefficients.
            # The product M_k (outer) is a 2x2 matrix of 2x2 blocks.
            # M_k = C_block.
            # C_block00 = M1' + M4' - M5' + M7'
            # C_block01 = M3' + M5'
            # C_block10 = M2' + M4'
            # C_block11 = M1' - M2' + M3' + M6'
            
            # The global C indices for block (bi, bj) are (2*bi + li, 2*bj + lj).
            # But W is indexed by (k, i) where k is col, i is row.
            # So C[i, k] -> W[k, i].
            # Block (bi, bj) of C corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1.
            
            for r in range(7):
                u_r = inner_u[r]
                v_r = inner_v[r]
                
                # Determine W coefficients
                w_r = {}
                
                # M1' contributes to C00 and C11
                # M2' contributes to C10
                # M3' contributes to C01 and C11
                # M4' contributes to C00 and C10
                # M5' contributes to C00 and C01
                # M6' contributes to C11
                # M7' contributes to C00 and C11
                
                if r == 0: # M1'
                    # C00: +1
                    for li in range(2):
                        for lj in range(2):
                            # C row: 0*2+li, C col: 0*2+lj
                            add_to_dict(w_r, (0*2+lj, 0*2+li), 1) # W[col, row]
                    # C11: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 1*2+li), 1)
                elif r == 1: # M2'
                    # C10: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (0*2+lj, 1*2+li), 1)
                elif r == 2: # M3'
                    # C01: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 0*2+li), 1)
                    # C11: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 1*2+li), 1)
                elif r == 3: # M4'
                    # C00: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (0*2+lj, 0*2+li), 1)
                    # C10: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (0*2+lj, 1*2+li), 1)
                elif r == 4: # M5'
                    # C00: -1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (0*2+lj, 0*2+li), -1)
                    # C01: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 0*2+li), 1)
                elif r == 5: # M6'
                    # C11: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 1*2+li), 1)
                elif r == 6: # M7'
                    # C00: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (0*2+lj, 0*2+li), 1)
                    # C11: +1
                    for li in range(2):
                        for lj in range(2):
                            add_to_dict(w_r, (1*2+lj, 1*2+li), 1)
                
                add_term(u_r, v_r, w_r)
                
        return U_terms, V_terms, W_terms

    U_terms, V_terms, W_terms = generate_49_terms()
    
    # Now handle the 4x4x1 part (k=4)
    # This is just the schoolbook for the 5th column.
    # A is 4x4, B_col is 4x1, C_col is 4x1.
    # C_i4 = sum_j A_ij B_j4.
    # Terms: for each i, j: U[i, j] = 1, V[j, 4] = 1, W[4, i] = 1.
    
    for i in range(4):
        for j in range(4):
            u_term = {}
            v_term = {}
            w_term = {}
            
            add_to_dict(u_term, (i, j), 1)
            add_to_dict(v_term, (j, 4), 1)
            add_to_dict(w_term, (4, i), 1)
            
            U_terms.append(u_term)
            V_terms.append(v_term)
            W_terms.append(w_term)
            
    # Convert to arrays
    rank = len(U_terms)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (i, j), coeff in U_terms[r].items():
            U[i * m + j, r] = coeff
        for (j, k), coeff in V_terms[r].items():
            V[j * p + k, r] = coeff
        for (k, i), coeff in W_terms[r].items():
            W[k * n + i, r] = coeff
            
    # Verify
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = get_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

