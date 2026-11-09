import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # We will build U, V, W as lists of lists.
    # Total rank = 49 (for 4x4x4) + 80 (for 4x4x5) = 129.
    R = 129
    U = [[0.0] * R for _ in range(n * m)]
    V = [[0.0] * R for _ in range(m * p)]
    W = [[0.0] * R for _ in range(p * n)]
    
    # Helper to add a rank-1 term
    # u_idx, u_val: index in U, value
    # v_idx, v_val: index in V, value
    # w_idx, w_val: index in W, value
    # col: column index in U, V, W
    def add_term(col, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        for idx, val in zip(u_indices, u_vals):
            U[idx][col] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][col] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][col] += val

    # Strassen 2x2 coefficients
    # Indices for 2x2:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
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
    
    strassen_2x2 = [
        # (u_coeffs, v_coeffs, w_coeffs)
        # u: indices [0,1,2,3], v: [0,1,2,3], w: [0,1,2,3]
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [0, -1, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Now build the 4x4x4 decomposition
    # A blocks: A0 (0,0), A1 (0,1), A2 (1,0), A3 (1,1)
    # In 4x4 index:
    # A0: 0,1,4,5
    # A1: 2,3,6,7
    # A2: 8,9,12,13
    # A3: 10,11,14,15
    
    # B blocks: B0 (0,0), B1 (0,1), B2 (1,0), B3 (1,1)
    # In 4x4 index (4x4):
    # B0: 0,1,4,5
    # B1: 2,3,6,7
    # B2: 8,9,12,13
    # B3: 10,11,14,15
    
    # C blocks: C0 (0,0), C1 (0,1), C2 (1,0), C3 (1,1)
    # In 4x4 index (4x4):
    # C0: 0,1,4,5
    # C1: 2,3,6,7
    # C2: 8,9,12,13
    # C3: 10,11,14,15
    
    # We need to map the 2x2 indices to 4x4 indices for each block.
    # For A:
    # Block 0 (A00): 2x2 indices 0,1,2,3 map to 4x2 indices 0,1,4,5
    # Block 1 (A01): 2x2 indices 0,1,2,3 map to 4x2 indices 2,3,6,7
    # Block 2 (A10): 2x2 indices 0,1,2,3 map to 4x2 indices 8,9,12,13
    # Block 3 (A11): 2x2 indices 0,1,2,3 map to 4x2 indices 10,11,14,15
    
    def map_2x2_to_4x2(block, coeff):
        # block: 0, 1, 2, 3
        # coeff: list of 4 values
        # returns list of (index, value)
        res = []
        for k, val in enumerate(coeff):
            if val != 0:
                # k is 0,1,2,3
                # row = k // 2, col = k % 2
                r = k // 2
                c = k % 2
                # block determines offset
                # A blocks:
                # 0: r*4 + c*2? No.
                # A is 4x4.
                # Block 0 is rows 0-1, cols 0-1.
                # Block 1 is rows 0-1, cols 2-3.
                # Block 2 is rows 2-3, cols 0-1.
                # Block 3 is rows 2-3, cols 2-3.
                
                if block == 0:
                    idx = r * 4 + c
                elif block == 1:
                    idx = r * 4 + (c + 2)
                elif block == 2:
                    idx = (r + 2) * 4 + c
                elif block == 3:
                    idx = (r + 2) * 4 + (c + 2)
                res.append((idx, val))
        return res

    # Same mapping for B and C (both 4x4)
    
    # Strassen 4x4 coefficients
    # M1 = (A0+A3)(B0+B3)
    # M2 = (A2+A3)B0
    # M3 = A0(B1-B3)
    # M4 = A3(B2-B0)
    # M5 = (A0+A1)B3
    # M6 = (A2-A0)(B0+B1)
    # M7 = (A1-A3)(B2+B3)
    
    # C0 = M1 + M4 - M5 + M7
    # C1 = M3 + M5
    # C2 = M2 + M4
    # C3 = M1 - M2 + M3 + M6
    
    strassen_4x4_meta = [
        # (u_blocks, v_blocks, w_blocks)
        # u_blocks: dict block -> coeff
        # v_blocks: dict block -> coeff
        # w_blocks: dict block -> coeff
        # M1
        ({0: [1,0,0,1], 3: [1,0,0,1]}, {0: [1,0,0,1], 3: [1,0,0,1]}, {0: [1,0,0,1], 3: [1,0,0,1]}),
        # M2
        ({2: [0,0,1,1], 3: [0,0,1,1]}, {0: [1,0,0,0]}, {2: [0,0,1,0]}),
        # M3
        ({0: [1,0,0,0]}, {1: [0,1,0,-1], 3: [0,-1,0,1]}, {1: [0,1,0,1], 3: [0,1,0,1]}),
        # M4
        ({3: [0,0,0,1]}, {2: [0,-1,1,0], 0: [0,1,-1,0]}, {0: [1,0,1,0], 2: [1,0,1,0]}),
        # M5
        ({0: [1,1,0,0], 1: [1,1,0,0]}, {3: [0,0,0,1]}, {0: [-1,1,0,0], 3: [-1,1,0,0]}),
        # M6
        ({2: [1,0,-1,0], 0: [-1,0,1,0]}, {0: [1,1,0,0], 1: [1,1,0,0]}, {3: [0,0,0,1]}),
        # M7
        ({1: [0,1,0,-1], 3: [0,-1,0,1]}, {2: [0,0,1,1], 3: [0,0,1,1]}, {0: [1,0,0,0]}),
    ]
    
    col = 0
    for meta_u, meta_v, meta_w in strassen_4x4_meta:
        # For each of the 7 scalar products in Strassen 2x2
        for u_2, v_2, w_2 in strassen_2x2:
            u_indices = []
            u_vals = []
            v_indices = []
            v_vals = []
            w_indices = []
            w_vals = []
            
            # Combine meta and 2x2 coefficients
            # meta_u is a dict of block -> 2x2 coeff
            # u_2 is a 2x2 coeff
            # The product is element-wise?
            # No.
            # M_l = U_l V_l.
            # U_l is a 4x4 matrix.
            # U_l is constructed by placing the 2x2 matrix (coeff from meta_u) into the blocks.
            # But wait.
            # M_l = (A_comb) * (B_comb).
            # A_comb is a 4x4 matrix.
            # The tensor for M_l is the tensor for the product of two 4x4 matrices?
            # No, M_l is just an intermediate result.
            # We are decomposing the tensor for C = AB.
            # C = sum of combinations of M_l.
            # Each M_l is a product of two 4x4 matrices?
            # No, in Strassen, M_l is a product of two 2x2 matrices if we are doing 4x4.
            # Wait.
            # If we do Strassen for 4x4, we split into 2x2 blocks.
            # M1 = (A11+A22)(B11+B22).
            # A11+A22 is a 2x2 matrix. B11+B22 is a 2x2 matrix.
            # So M1 is a 2x2 matrix.
            # To compute M1, we need to multiply two 2x2 matrices.
            # We can use Strassen for this 2x2 multiplication (rank 7).
            # So for each M_l, we add 7 rank-1 terms.
            
            # The 2x2 matrix for A_part is formed by combining the blocks.
            # e.g. for M1, A_part = A0 + A3.
            # The coefficient for A0 is 1, for A3 is 1.
            # So the 2x2 matrix A_part has entries A0_ij + A3_ij.
            # In the tensor decomposition, we need to express A_part as a linear combination of the original A entries.
            # This is handled by the meta_u coefficients.
            
            # However, the rank-1 terms for the 2x2 multiplication of A_part and B_part
            # act on the entries of A_part and B_part.
            # So we need to map the 2x2 indices of A_part back to the 4x4 indices of A.
            
            # Let's refine the logic.
            # For a specific M_l:
            # We have A_part = sum_{b} c_b A_b.
            # We have B_part = sum_{b} d_b B_b.
            # M_l = A_part * B_part.
            # We decompose the 2x2 multiplication A_part * B_part using Strassen 2x2.
            # This gives 7 rank-1 terms for the 2x2 tensor.
            # Each term is u_2 * v_2 * w_2.
            # u_2 is a 2x2 matrix.
            # We need to expand u_2 into a 4x4 matrix U_l by placing u_2 entries into the blocks.
            # Specifically, if u_2 has entry $u_{ij}$ at 2x2 index (i,j),
            # and the block b has coefficient c_b in A_part,
            # then the 4x4 matrix U_l has entry $c_b u_{ij}$ at the position corresponding to block b, index (i,j).
            
            # So:
            # For each block b in meta_u:
            #   coeff_b = meta_u[b] (a 2x2 matrix of coefficients)
            #   For each 2x2 index (i,j) in coeff_b:
            #     val_b = coeff_b[i,j]
            #     For each 2x2 index (k,l) in u_2:
            #       val_u = u_2[k,l]
            #       The contribution to U_l at 4x4 index (block b, pos k,l) is val_b * val_u.
            
            # This is equivalent to:
            # U_l = sum_b (coeff_b \otimes u_2) placed in block b?
            # No, coeff_b is a 2x2 matrix. u_2 is a 2x2 matrix.
            # The product is not defined simply.
            # Wait.
            # A_part is a 2x2 matrix.
            # A_part = c0 * A0 + c1 * A1 ... ?
            # No, A0, A1 are 2x2 matrices.
            # A_part = A0 + A3.
            # So A_part is a 2x2 matrix whose entries are sums of entries from A0 and A3.
            # The "coefficient" of A0 is the identity matrix I_2.
            # The "coefficient" of A3 is the identity matrix I_2.
            
            # So meta_u should be a dict of block -> 2x2 matrix of coefficients.
            # For M1, meta_u = {0: I, 3: I}.
            
            # Then U_l (4x4) is constructed by:
            # For each block b:
            #   C_b = meta_u[b] (2x2 matrix)
            #   The block of U_l at position b is C_b * u_2?
            # No.
            # A_part is a 2x2 matrix.
            # The entries of A_part are linear combinations of entries of A.
            # Let A_part[i,j] = sum_b C_b[i,j] * A_b[i,j].
            # Then M_l = A_part * B_part.
            # The tensor for M_l is T_{i,j,k} = A_part[i,j] B_part[j,k] ...
            # We decompose A_part * B_part using Strassen 2x2.
            # The Strassen 2x2 decomposition gives rank-1 terms u_2, v_2, w_2.
            # These u_2, v_2 are 2x2 matrices.
            # The term is sum u_2[i,j] A_part[i,j] * v_2[j,k] B_part[j,k] * w_2[i,k] M_l[i,k].
            # Substitute A_part and B_part:
            # sum u_2[i,j] (sum_b C_b[i,j] A_b[i,j]) * v_2[j,k] (sum_d D_d[j,k] B_d[j,k]) * w_2[i,k] M_l[i,k].
            # = sum_b sum_d C_b[i,j] D_d[j,k] u_2[i,j] v_2[j,k] w_2[i,k] A_b[i,j] B_d[j,k] M_l[i,k].
            
            # So the 4x4 matrix U_l has entry at (block b, pos i,j) equal to C_b[i,j] * u_2[i,j].
            # Similarly for V_l and W_l.
            
            # Let's implement this.
            
            for b, c_b in meta_u.items():
                # c_b is a 2x2 matrix (list of lists or flat)
                # We stored it as flat [1,0,0,1]
                for i in range(2):
                    for j in range(2):
                        idx_2x2 = i * 2 + j
                        val_c = c_b[idx_2x2]
                        if val_c == 0: continue
                        # u_2 is flat [u00, u01, u10, u11]
                        val_u = u_2[idx_2x2]
                        if val_u == 0: continue
                        val_total = val_c * val_u
                        # Map to 4x4 index
                        idx_4x4 = map_2x2_to_4x2(b, [0,0,0,0]) # dummy
                        # Actually map_2x2_to_4x2 takes coeff.
                        # Let's just compute index directly.
                        r = i
                        c = j
                        if b == 0: idx = r * 4 + c
                        elif b == 1: idx = r * 4 + (c + 2)
                        elif b == 2: idx = (r + 2) * 4 + c
                        elif b == 3: idx = (r + 2) * 4 + (c + 2)
                        u_indices.append(idx)
                        u_vals.append(val_total)
                        
            for b, c_b in meta_v.items():
                for i in range(2):
                    for j in range(2):
                        idx_2x2 = i * 2 + j
                        val_c = c_b[idx_2x2]
                        if val_c == 0: continue
                        val_v = v_2[idx_2x2]
                        if val_v == 0: continue
                        val_total = val_c * val_v
                        r = i
                        c = j
                        if b == 0: idx = r * 4 + c
                        elif b == 1: idx = r * 4 + (c + 2)
                        elif b == 2: idx = (r + 2) * 4 + c
                        elif b == 3: idx = (r + 2) * 4 + (c + 2)
                        v_indices.append(idx)
                        v_vals.append(val_total)
                        
            for b, c_b in meta_w.items():
                for i in range(2):
                    for j in range(2):
                        idx_2x2 = i * 2 + j
                        val_c = c_b[idx_2x2]
                        if val_c == 0: continue
                        val_w = w_2[idx_2x2]
                        if val_w == 0: continue
                        val_total = val_c * val_w
                        r = i
                        c = j
                        if b == 0: idx = r * 4 + c
                        elif b == 1: idx = r * 4 + (c + 2)
                        elif b == 2: idx = (r + 2) * 4 + c
                        elif b == 3: idx = (r + 2) * 4 + (c + 2)
                        w_indices.append(idx)
                        w_vals.append(val_total)
                        
            add_term(col, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)
            col += 1
            
    # Now add schoolbook for 4x4x5
    # This covers the last 5 columns of B (and C).
    # i in 0..3, j in 0..3, k in 4..8
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
