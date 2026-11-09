import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 4, 4, 5
    
    def time_remaining():
        return deadline - time.time()
    
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

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        if U_arr.ndim != 2 or V_arr.ndim != 2 or W_arr.ndim != 2:
            return False
        
        rank = U_arr.shape[1]
        if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
            return False
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True

    # Build rank-65 decomposition:
    # 49 terms from recursive Strassen for 4x4x4 (cols 0..3)
    # 16 terms for the 5th column (k=4)
    
    total_rank = 65
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    if time_remaining() < 5.0:
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Strassen 2x2: 7 multiplications
    # C = A * B where A, B are 2x2
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
    
    # For 4x4x4: partition into 2x2 blocks
    # A rows 0-1,2-3; cols 0-1,2-3
    # B rows 0-1,2-3; cols 0-1,2-3
    # C rows 0-1,2-3; cols 0-1,2-3
    
    # The 7 outer Strassen terms each produce a 2x2 result.
    # Each 2x2 multiplication uses 7 scalar multiplications (inner Strassen).
    # Total: 7 * 7 = 49 scalar multiplications.
    
    # We'll track each scalar multiplication as (u_combo, v_combo)
    # where u_combo is a dict {(i,j): coeff} for A entries
    # and v_combo is a dict {(i,j): coeff} for B entries
    # The W contribution comes from the C position.
    
    # Let's define the 7 Strassen multiplications for a 2x2 product.
    # Each returns (u_entries, v_entries) where entries are dicts.
    
    def strassen_2x2_mults():
        """
        Returns list of 7 tuples: (u_dict, v_dict)
        where u_dict maps (r,c) -> coeff for the 2x2 A matrix
        and v_dict maps (r,c) -> coeff for the 2x2 B matrix
        """
        mults = []
        # M1 = (A00+A11)(B00+B11)
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        mults.append((u1, v1))
        
        # M2 = (A10+A11)B00
        u2 = {(1,0): 1, (1,1): 1}
        v2 = {(0,0): 1}
        mults.append((u2, v2))
        
        # M3 = A00(B01-B11)
        u3 = {(0,0): 1}
        v3 = {(0,1): 1, (1,1): -1}
        mults.append((u3, v3))
        
        # M4 = A11(B10-B00)
        u4 = {(1,1): 1}
        v4 = {(1,0): 1, (0,0): -1}
        mults.append((u4, v4))
        
        # M5 = (A00+A01)B11
        u5 = {(0,0): 1, (0,1): 1}
        v5 = {(1,1): 1}
        mults.append((u5, v5))
        
        # M6 = (A10-A00)(B00+B01)
        u6 = {(1,0): 1, (0,0): -1}
        v6 = {(0,0): 1, (0,1): 1}
        mults.append((u6, v6))
        
        # M7 = (A01-A11)(B10+B11)
        u7 = {(0,1): 1, (1,1): -1}
        v7 = {(1,0): 1, (1,1): 1}
        mults.append((u7, v7))
        
        return mults
    
    # For each of the 7 outer Strassen terms, we have a 2x2 "effective" A and B.
    # These effective matrices are linear combinations of the 4x4 A and B blocks.
    # Then we apply inner Strassen to each 2x2 multiplication.
    
    # The 7 outer terms and their block definitions:
    # Let A_blocks: A00=(rows 0-1, cols 0-1), A01=(rows 0-1, cols 2-3),
    #                A10=(rows 2-3, cols 0-1), A11=(rows 2-3, cols 2-3)
    # Let B_blocks: B00=(rows 0-1, cols 0-1), B01=(rows 0-1, cols 2-3),
    #                B10=(rows 2-3, cols 0-1), B11=(rows 2-3, cols 2-3)
    
    # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
    # U1 is 2x2: U1[r,c] = A[r,c] + A[r+2,c+2] for r,c in {0,1}
    # V1 is 2x2: V1[r,c] = B[r,c] + B[r+2,c+2]
    
    # We need to express each entry of U1 and V1 as a linear combo of 4x4 entries.
    # Then apply inner Strassen.
    
    # Let me define a function that takes a 2x2 matrix defined as linear combos
    # of 4x4 entries, and returns the 7 inner Strassen scalar multiplications.
    
    def expand_2x2_to_scalars(u_matrix, v_matrix):
        """
        u_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 A entries
        v_matrix: 2x2 array where each entry is a dict {(i,j): coeff} of 4x4 B entries
        Returns list of 7 (u_scalar, v_scalar) where each is a dict {(i,j): coeff}
        """
        strassen_mults = strassen_2x2_mults()
        results = []
        for u_dict, v_dict in strassen_mults:
            u_scalar = {}
            v_scalar = {}
            for (ur, uc), coeff_u in u_dict.items():
                for (ai, aj), coeff_ai in u_matrix[ur][uc].items():
                    key = (ai, aj)
                    u_scalar[key] = u_scalar.get(key, 0) + coeff_u * coeff_ai
            for (vr, vc), coeff_v in v_dict.items():
                for (bi, bj), coeff_bi in v_matrix[vr][vc].items():
                    key = (bi, bj)
                    v_scalar[key] = v_scalar.get(key, 0) + coeff_v * coeff_bi
            # Remove zeros
            u_scalar = {k: v for k, v in u_scalar.items() if v != 0}
            v_scalar = {k: v for k, v in v_scalar.items() if v != 0}
            results.append((u_scalar, v_scalar))
        return results
    
    # Define the 7 outer terms as 2x2 matrices of linear combos
    
    def make_block_matrix(block_defs):
        """
        block_defs: list of 4 dicts, each {(i,j): coeff} for the 2x2 block
        Returns 2x2 array of dicts
        """
        return [[block_defs[r*2+c] for c in range(2)] for r in range(2)]
    
    # Outer term 1: U1 = A00 + A11, V1 = B00 + B11
    # A00[r,c] = A[r,c], A11[r,c] = A[r+2,c+2]
    U1_blocks = []
    for r in range(2):
        for c in range(2):
            U1_blocks.append({(r,c): 1, (r+2,c+2): 1})
    V1_blocks = []
    for r in range(2):
        for c in range(2):
            V1_blocks.append({(r,c): 1, (r+2,c+2): 1})
    
    # Outer term 2: U2 = A10 + A11, V2 = B00
    U2_blocks = []
    for r in range(2):
        for c in range(2):
            U2_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
    V2_blocks = []
    for r in range(2):
        for c in range(2):
            V2_blocks.append({(r,c): 1})
    
    # Outer term 3: U3 = A00, V3 = B01 - B11
    U3_blocks = []
    for r in range(2):
        for c in range(2):
            U3_blocks.append({(r,c): 1})
    V3_blocks = []
    for r in range(2):
        for c in range(2):
            V3_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
    
    # Outer term 4: U4 = A11, V4 = B10 - B00
    U4_blocks = []
    for r in range(2):
        for c in range(2):
            U4_blocks.append({(r+2,c+2): 1})
    V4_blocks = []
    for r in range(2):
        for c in range(2):
            V4_blocks.append({(r+2,c): 1, (r,c): -1})
    
    # Outer term 5: U5 = A00 + A01, V5 = B11
    U5_blocks = []
    for r in range(2):
        for c in range(2):
            U5_blocks.append({(r,c): 1, (r,c+2): 1})
    V5_blocks = []
    for r in range(2):
        for c in range(2):
            V5_blocks.append({(r+2,c+2): 1})
    
    # Outer term 6: U6 = A10 - A00, V6 = B00 + B01
    U6_blocks = []
    for r in range(2):
        for c in range(2):
            U6_blocks.append({(r+2,c): 1, (r,c): -1})
    V6_blocks = []
    for r in range(2):
        for c in range(2):
            V6_blocks.append({(r,c): 1, (r,c+2): 1})
    
    # Outer term 7: U7 = A01 - A11, V7 = B10 + B11
    U7_blocks = []
    for r in range(2):
        for c in range(2):
            U7_blocks.append({(r,c+2): 1, (r+2,c+2): -1})
    V7_blocks = []
    for r in range(2):
        for c in range(2):
            V7_blocks.append({(r+2,c): 1, (r+2,c+2): 1})
    
    # Now we need to determine the W contribution for each scalar multiplication.
    # The C matrix is computed from the 7 outer terms:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 matrix. The inner Strassen computes Mj = Uj * Vj.
    # The inner Strassen gives 7 scalar multiplications that combine to form Mj.
    # We need to know which C position each scalar multiplication contributes to.
    
    # Actually, the W tensor component W[k*n+i] for each scalar multiplication
    # is determined by which C entry it contributes to.
    
    # Let me think about this differently. The full 4x4x4 multiplication tensor
    # is: T[i,j,k] = sum_{a,b} A[i,a] * B[a,k] * delta_{j,b}
    # Wait, no. T[i,j,k] = sum_a A[i,a] * B[a,k] where the index j is the column of A.
    # Actually for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    # So the decomposition is: sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1 for all i,j,k
    # where U[i,j,r] * V[j,k,r] * W[i,k,r] should equal delta_{i,j,k} in some sense.
    # Actually, we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # The Strassen algorithm computes C = A*B. Each scalar multiplication s_r
    # contributes to certain entries of C. The W factor tells us which C entry
    # (i,k) this multiplication contributes to.
    
    # For the inner Strassen of Mj = Uj * Vj (2x2 result):
    # The 7 scalar multiplications produce the 4 entries of Mj.
    # Mj[0,0] = s1 + s4 - s5 + s7
    # Mj[0,1] = s3 + s5
    # Mj[1,0] = s2 + s4
    # Mj[1,1] = s1 - s2 + s3 + s6
    
    # So each scalar multiplication s_t contributes to specific positions of Mj.
    # And Mj contributes to specific positions of C.
    
    # Let me define the contribution of each inner Strassen multiplication
    # to the 2x2 result matrix.
    
    # Inner Strassen multiplication t contributes to Mj positions:
    inner_contrib = {
        0: [(0,0, 1), (1,1, 1)],      # s1 -> M[0,0] + M[1,1]
        1: [(1,0, 1), (1,1, -1)],     # s2 -> M[1,0] - M[1,1]
        2: [(0,1, 1), (1,1, 1)],      # s3 -> M[0,1] + M[1,1]
        3: [(0,0, 1), (1,0, 1)],      # s4 -> M[0,0] + M[1,0]
        4: [(0,0, -1), (0,1, 1)],     # s5 -> -M[0,0] + M[0,1]
        5: [(1,1, 1)],                # s6 -> M[1,1]
        6: [(0,0, 1)],                # s7 -> M[0,0]
    }
    
    # Outer term j contributes to C positions:
    outer_contrib = {
        0: [(0,0, 1), (1,1, 1)],      # M1 -> C[0,0] + C[1,1]
        1: [(1,0, 1), (1,1, -1)],     # M2 -> C[1,0] - C[1,1]
        2: [(0,1, 1), (1,1, 1)],      # M3 -> C[0,1] + C[1,1]
        3: [(0,0, 1), (1,0, 1)],      # M4 -> C[0,0] + C[1,0]
        4: [(0,0, -1), (0,1, 1)],     # M5 -> -C[0,0] + C[0,1]
        5: [(1,1, 1)],                # M6 -> C[1,1]
        6: [(0,0, 1)],                # M7 -> C[0,0]
    }
    
    # Now, for each outer term j and inner multiplication t:
    # The scalar multiplication s = u_scalar * v_scalar
    # contributes to C positions: for each (mr, mc, coeff_inner) in inner_contrib[t]
    #   and each (cr, cc, coeff_outer) in outer_contrib[j]:
    #     C position (cr, cc) gets coeff_inner * coeff_outer
    # But wait, Mj is a 2x2 block of C. Mj[mr,mc] corresponds to C[2*mr+? , 2*mc+?]
    # No, Mj is the entire 2x2 result. The outer Strassen combines the 7 Mj's
    # to form the 4x4 C. Each Mj is a 2x2 matrix that gets added to specific
    # 2x2 blocks of C.
    
    # Actually, I think I'm overcomplicating this. Let me reconsider.
    
    # In Strassen for 4x4:
    # C is 4x4, partitioned into four 2x2 blocks: C00, C01, C10, C11
    # C00 = M1 + M4 - M5 + M7  (all 2x2 matrices)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Mj is a 2x2 matrix computed as Uj * Vj using inner Strassen.
    # The inner Strassen produces 7 scalar multiplications that combine to form Mj.
    
    # So for C00 = M1 + M4 - M5 + M7:
    # Each 2x2 entry of C00 is a sum of corresponding entries of M1, M4, M5, M7.
    # Each entry of Mj is a sum of the 7 inner scalar multiplications.
    
    # So C00[0,0] = M1[0,0] + M4[0,0] - M5[0,0] + M7[0,0]
    #            = (s1_1 + s4_1 - s5_1 + s7_1) + (s1_4 + s4_4 - s5_4 + s7_4) 
    #              - (s1_5 + s4_5 - s5_5 + s7_5) + (s1_7 + s4_7 - s5_7 + s7_7)
    # where s_t_j is the t-th inner Strassen multiplication of outer term j.
    
    # This means each scalar multiplication s_t_j contributes to multiple C entries.
    # The W factor for this scalar multiplication should have non-zero entries
    # at all C positions it contributes to.
    
    # Let me compute the W contribution for each (j, t) pair.
    
    # C block positions: C00 is rows 0-1, cols 0-1; C01 is rows 0-1, cols 2-3;
    # C10 is rows 2-3, cols 0-1; C11 is rows 2-3, cols 2-3
    
    # For outer term j, the 2x2 Mj contributes to C blocks as follows:
    # j=0 (M1): C00 coeff 1, C11 coeff 1
    # j=1 (M2): C10 coeff 1, C11 coeff -1
    # j=2 (M3): C01 coeff 1, C11 coeff 1
    # j=3 (M4): C00 coeff 1, C10 coeff 1
    # j=4 (M5): C00 coeff -1, C01 coeff 1
    # j=5 (M6): C11 coeff 1
    # j=6 (M7): C00 coeff 1
    
    # For inner multiplication t of Mj, the contribution to Mj's 2x2 entries:
    # t=0: Mj[0,0] coeff 1, Mj[1,1] coeff 1
    # t=1: Mj[1,0] coeff 1, Mj[1,1] coeff -1
    # t=2: Mj[0,1] coeff 1, Mj[1,1] coeff 1
    # t=3: Mj[0,0] coeff 1, Mj[1,0] coeff 1
    # t=4: Mj[0,0] coeff -1, Mj[0,1] coeff 1
    # t=5: Mj[1,1] coeff 1
    # t=6: Mj[0,0] coeff 1
    
    # So for (j, t), the C positions and coefficients:
    # For each (block, block_coeff) in outer_contrib[j]:
    #   For each (mr, mc, inner_coeff) in inner_contrib[t]:
    #     C row = block_row_offset + mr
    #     C col = block_col_offset + mc
    #     coeff = block_coeff * inner_coeff
    
    # Block offsets:
    block_offsets = {
        'C00': (0, 0),
        'C01': (0, 2),
        'C10': (2, 0),
        'C11': (2, 2),
    }
    
    outer_block_map = {
        0: [('C00', 1), ('C11', 1)],
        1: [('C10', 1), ('C11', -1)],
        2: [('C01', 1), ('C11', 1)],
        3: [('C00', 1), ('C10', 1)],
        4: [('C00', -1), ('C01', 1)],
        5: [('C11', 1)],
        6: [('C00', 1)],
    }
    
    # Now generate all 49 scalar multiplications
    outer_terms = [
        (U1_blocks, V1_blocks),
        (U2_blocks, V2_blocks),
        (U3_blocks, V3_blocks),
        (U4_blocks, V4_blocks),
        (U5_blocks, V5_blocks),
        (U6_blocks, V6_blocks),
        (U7_blocks, V7_blocks),
    ]
    
    col_idx = 0
    for j in range(7):
        U_blocks, V_blocks = outer_terms[j]
        # Convert to 2x2 arrays of dicts
        U_mat = [[U_blocks[r*2+c] for c in range(2)] for r in range(2)]
        V_mat = [[V_blocks[r*2+c] for c in range(2)] for r in range(2)]
        
        inner_mults = expand_2x2_to_scalars(U_mat, V_mat)
        
        for t in range(7):
            u_scalar, v_scalar = inner_mults[t]
            
            # Compute W contributions
            w_dict = {}
            for block_name, block_coeff in outer_block_map[j]:
                row_off, col_off = block_offsets[block_name]
                for mr, mc, inner_coeff in inner_contrib[t]:
                    c_row = row_off + mr
                    c_col = col_off + mc
                    coeff = block_coeff * inner_coeff
                    # W index: k*n + i where k is C col, i is C row
                    # Wait, W[k*n+i, r] where C[i,k] = sum_j A[i,j]*B[j,k]
                    # So W index for C[i,k] is k*n + i
                    w_idx = c_col * n + c_row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + coeff
            
            # Fill in U, V, W for this column
            for (ai, aj), coeff in u_scalar.items():
                U[ai * m + aj, col_idx] += coeff
            
            for (bi, bj), coeff in v_scalar.items():
                V[bi * p + bj, col_idx] += coeff
            
            for w_idx, coeff in w_dict.items():
                W[w_idx, col_idx] += coeff
            
            col_idx += 1
    
    # Now add 16 scalar multiplications for the 5th column (k=4)
    # C[i,4] = sum_j A[i,j] * B[j,4]
    # For each i in 0..3, j in 0..3: one scalar multiplication
    for i in range(n):
        for j in range(m):
            U[i * m + j, col_idx] = 1.0
            V[j * p + 4, col_idx] = 1.0
            W[4 * n + i, col_idx] = 1.0
            col_idx += 1
    
    # Verify
    if time_remaining() < 3.0:
        # Not enough time to verify, return as-is (should be correct)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = get_schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
