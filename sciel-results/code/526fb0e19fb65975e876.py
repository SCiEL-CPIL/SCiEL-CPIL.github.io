import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=6
    
    # We will build the decomposition in two parts:
    # Part 1: 4x4 by 4x4 (first 4 columns of B) using Strassen (Rank 49)
    # Part 2: 4x4 by 4x2 (last 2 columns of B) using Schoolbook (Rank 32)
    
    rank1 = 49
    rank2 = 32
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: Strassen 4x4x4 ---
    # Indices for U: i in 0..3, j in 0..3
    # Indices for V: j in 0..3, k in 0..3 (only first 4 cols)
    # Indices for W: k in 0..3, i in 0..3
    
    # We need the 7x7 decomposition of 4x4x4.
    # This is derived from the 7x7 decomposition of 2x2x2.
    
    # Let's define the 2x2x2 Strassen factors.
    # U_2x2 shape (4, 7), V_2x2 shape (4, 7), W_2x2 shape (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # The 7 products are defined by linear combinations of inputs.
    # Instead of hardcoding 49 entries, we can generate them.
    
    # Strassen 2x2x2 factors (rank 7)
    # Each factor r in 0..6 corresponds to a product of two linear forms.
    # Let's define the coefficients for U, V, W for each of the 7 factors.
    
    # Factor 0: (A00+A11)(B00+B11)
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00+=1, C01+=1, C10+=1, C11+=1 (Wait, W is output)
    # W[k*2+i]
    # M1 contributes to all 4 outputs with coeff 1.
    
    # It is safer to just use the standard algorithm to generate the factors for 2x2x2 and then compose.
    
    def get_strassen_2x2x2():
        # Returns U, V, W for 2x2x2
        # U (4x7), V (4x7), W (4x7)
        u = [[0]*7 for _ in range(4)]
        v = [[0]*7 for _ in range(4)]
        w = [[0]*7 for _ in range(4)]
        
        # Factor 0: M1 = (A00+A11)(B00+B11)
        # A00 is U[0], A11 is U[3]
        u[0][0] = 1; u[3][0] = 1
        # B00 is V[0], B11 is V[3]
        v[0][0] = 1; v[3][0] = 1
        # C00 (W[0]), C01 (W[2]), C10 (W[1]), C11 (W[3])
        # W index: k*2+i.
        # C00: k=0,i=0 -> W[0]
        # C01: k=1,i=0 -> W[2]
        # C10: k=0,i=1 -> W[1]
        # C11: k=1,i=1 -> W[3]
        w[0][0] = 1; w[2][0] = 1; w[1][0] = 1; w[3][0] = 1
        
        # Factor 1: M2 = (A10+A11)B00
        # A10 is U[2], A11 is U[3]
        u[2][1] = 1; u[3][1] = 1
        # B00 is V[0]
        v[0][1] = 1
        # C10 (W[1]), C11 (W[3])
        w[1][1] = 1; w[3][1] = 1
        
        # Factor 2: M3 = A00(B01-B11)
        # A00 is U[0]
        u[0][2] = 1
        # B01 is V[1], B11 is V[3]
        v[1][2] = 1; v[3][2] = -1
        # C01 (W[2]), C11 (W[3])
        w[2][2] = 1; w[3][2] = 1
        
        # Factor 3: M4 = A11(B10-B00)
        # A11 is U[3]
        u[3][3] = 1
        # B10 is V[2], B00 is V[0]
        v[2][3] = 1; v[0][3] = -1
        # C10 (W[1]), C11 (W[3])
        w[1][3] = 1; w[3][3] = 1
        
        # Factor 4: M5 = (A00+A01)B11
        # A00 is U[0], A01 is U[1]
        u[0][4] = 1; u[1][4] = 1
        # B11 is V[3]
        v[3][4] = 1
        # C01 (W[2]), C11 (W[3])
        w[2][4] = 1; w[3][4] = 1
        
        # Factor 5: M6 = (A10-A00)(B00+B01)
        # A10 is U[2], A00 is U[0]
        u[2][5] = 1; u[0][5] = -1
        # B00 is V[0], B01 is V[1]
        v[0][5] = 1; v[1][5] = 1
        # C10 (W[1]), C11 (W[3])
        w[1][5] = 1; w[3][5] = 1
        
        # Factor 6: M7 = (A01-A11)(B10+B11)
        # A01 is U[1], A11 is U[3]
        u[1][6] = 1; u[3][6] = -1
        # B10 is V[2], B11 is V[3]
        v[2][6] = 1; v[3][6] = 1
        # C01 (W[2]), C11 (W[3])
        w[2][6] = 1; w[3][6] = 1
        
        return u, v, w

    u2, v2, w2 = get_strassen_2x2x2()
    
    # Now build 4x4x4 using 2x2x2 blocks.
    # The 4x4x4 decomposition has 7 "super-factors", each of which is a 2x2x2 multiplication.
    # Each 2x2x2 multiplication has 7 rank-1 terms.
    # So total 49 terms.
    
    # We iterate over the 7 super-factors (sf in 0..6)
    # For each sf, we identify which 2x2 blocks of A and B are involved, and how they are combined.
    # Then we iterate over the 7 rank-1 terms of the 2x2x2 decomposition.
    
    # Mapping of super-factors to block operations:
    # sf 0: M1 = (A00+A11)(B00+B11)
    # sf 1: M2 = (A10+A11)B00
    # sf 2: M3 = A00(B01-B11)
    # sf 3: M4 = A11(B10-B00)
    # sf 4: M5 = (A00+A01)B11
    # sf 5: M6 = (A10-A00)(B00+B01)
    # sf 6: M7 = (A01-A11)(B10+B11)
    
    # Block indices:
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # For each super-factor, we need to know:
    # - The linear combination of A blocks (coeffs for A00, A01, A10, A11)
    # - The linear combination of B blocks (coeffs for B00, B01, B10, B11)
    # - The linear combination of C blocks (coeffs for C00, C01, C10, C11)
    
    # Let's define these explicitly.
    # a_coeffs: dict block_index -> coeff
    # b_coeffs: dict block_index -> coeff
    # c_coeffs: dict block_index -> coeff
    
    super_factors = [
        # sf 0
        {'a': {0: 1, 3: 1}, 'b': {0: 1, 3: 1}, 'c': {0: 1, 1: 1, 2: 1, 3: 1}},
        # sf 1
        {'a': {2: 1, 3: 1}, 'b': {0: 1}, 'c': {2: 1, 3: 1}},
        # sf 2
        {'a': {0: 1}, 'b': {1: 1, 3: -1}, 'c': {1: 1, 3: 1}},
        # sf 3
        {'a': {3: 1}, 'b': {2: 1, 0: -1}, 'c': {2: 1, 3: 1}},
        # sf 4
        {'a': {0: 1, 1: 1}, 'b': {3: 1}, 'c': {1: 1, 3: 1}},
        # sf 5
        {'a': {2: 1, 0: -1}, 'b': {0: 1, 1: 1}, 'c': {2: 1, 3: 1}},
        # sf 6
        {'a': {1: 1, 3: -1}, 'b': {2: 1, 3: 1}, 'c': {1: 1, 3: 1}},
    ]
    
    # Block 00 corresponds to rows 0-1, cols 0-1
    # Block 01 corresponds to rows 0-1, cols 2-3
    # Block 10 corresponds to rows 2-3, cols 0-1
    # Block 11 corresponds to rows 2-3, cols 2-3
    
    # For A (4x4), block bi (0..3) has rows 2*(bi//2) .. 2*(bi//2)+1 and cols 2*(bi%2) .. 2*(bi%2)+1
    # For B (4x4), block bi (0..3) has rows 2*(bi//2) .. 2*(bi//2)+1 and cols 2*(bi%2) .. 2*(bi%2)+1
    # For C (4x4), block bi (0..3) has rows 2*(bi//2) .. 2*(bi//2)+1 and cols 2*(bi%2) .. 2*(bi%2)+1
    
    col_offset = 0
    for sf_idx, sf in enumerate(super_factors):
        for r2 in range(7):
            # r2 is the rank-1 term index in the 2x2x2 decomposition
            # We need to add the contributions of u2, v2, w2 scaled by the block coefficients
            
            # Iterate over all elements in the 2x2x2 factors
            # u2 has shape 4x7. v2 has shape 4x7. w2 has shape 4x7.
            
            # We need to map the local 2x2 indices to global 4x4 indices.
            
            # For U (A):
            # u2[local_i, r2] is non-zero.
            # local_i corresponds to row li_r, col li_c in the 2x2 block.
            # The 2x2 block is determined by the super-factor's A block combination.
            # Wait, the super-factor combines A blocks.
            # E.g., sf 0 uses A00 and A11.
            # The "input" to the 2x2x2 multiplication is a linear combination of A blocks.
            # Let A_block_comb be a 2x2 matrix where each element is a linear combo of A block elements.
            # Actually, it's simpler:
            # The super-factor computes M = (sum a_coeff[bi] * A_bi) * (sum b_coeff[bi] * B_bi).
            # This M is a 2x2 matrix.
            # Then we decompose M = C_comb (sum c_coeff[bi] * C_bi).
            
            # So, for each local element (li, lj) in the 2x2 input A:
            # The value is sum over bi of a_coeff[bi] * A_bi[li, lj]
            # A_bi[li, lj] corresponds to global A[2*(bi//2)+li, 2*(bi%2)+lj]
            
            # Similarly for B.
            
            # And for C:
            # The value is sum over bi of c_coeff[bi] * C_bi[li, lj]
            # C_bi[li, lj] corresponds to global C[2*(bi//2)+li, 2*(bi%2)+lj]
            
            # So, for each rank-1 term r2:
            # We have coefficients u2[li, r2], v2[lj, r2], w2[lo, r2] (where lo is local C index)
            # Wait, u2 index is li (row of A), v2 index is lj (col of A / row of B), w2 index is lo (col of B / row of C)
            # Actually, standard tensor notation:
            # U[i, r], V[j, r], W[k, r]
            # C[i, k] = sum_r U[i, r] V[j, r] W[k, r] ? No.
            # C[i, k] = sum_j A[i, j] B[j, k]
            # Tensor T[i, j, k] = delta.
            # U[i, r] V[j, r] W[k, r] sums to T[i, j, k].
            
            # In our 2x2x2 decomposition:
            # u2[local_i, r2] is coeff for A[local_i, local_j]
            # v2[local_j, r2] is coeff for B[local_j, local_k]
            # w2[local_k, r2] is coeff for C[local_i, local_k]
            
            # So for each r2:
            # For each local_i, local_j, local_k:
            # coeff_u = u2[local_i, r2]
            # coeff_v = v2[local_j, r2]
            # coeff_w = w2[local_k, r2]
            
            # We need to expand the block combinations.
            
            # Let's precompute the global indices for each local index and block.
            
            for li in range(2):
                for lj in range(2):
                    for lk in range(2):
                        cu = u2[li, r2]
                        cv = v2[lj, r2]
                        cw = w2[lk, r2]
                        
                        if cu == 0 or cv == 0 or cw == 0:
                            continue
                        
                        # Expand A part
                        # Sum over bi in sf['a']
                        # Global U index: i_global * m + j_global
                        # i_global = 2*(bi//2) + li
                        # j_global = 2*(bi%2) + lj
                        
                        # Expand B part
                        # Global V index: j_global * p + k_global
                        # j_global = 2*(bi//2) + lj  (Note: B row index is same as A col index in this block structure?
                        # Yes, A is 4x4, B is 4x4. A[i,j] B[j,k].
                        # Block bi of A has cols 2*(bi%2)..2*(bi%2)+1.
                        # Block bi of B has rows 2*(bi//2)..2*(bi//2)+1.
                        # Wait, the block index bi for A and B must match for the product to be valid in Strassen?
                        # Yes, Strassen operates on corresponding blocks.
                        # A00 * B00, etc.
                        
                        # Expand C part
                        # Global W index: k_global * n + i_global
                        # k_global = 2*(bi//2) + lk
                        # i_global = 2*(bi%2) + li  (Wait, C block bi has rows 2*(bi//2).. and cols 2*(bi%2)..
                        # C[i, k]. i is row, k is col.
                        # W index is k*n + i.
                        # So i_global is row of C, k_global is col of C.
                        # C block bi: rows 2*(bi//2)+li, cols 2*(bi%2)+lk.
                        
                        # So:
                        # i_A = 2*(bi_a//2) + li
                        # j_A = 2*(bi_a%2) + lj
                        # j_B = 2*(bi_b//2) + lj
                        # k_B = 2*(bi_b%2) + lk
                        # i_C = 2*(bi_c//2) + li
                        # k_C = 2*(bi_c%2) + lk
                        
                        # In Strassen, the block indices for A, B, C are linked.
                        # Specifically, the super-factor sf uses specific blocks.
                        # For A, it uses blocks in sf['a'].
                        # For B, it uses blocks in sf['b'].
                        # For C, it uses blocks in sf['c'].
                        
                        # But the local indices li, lj, lk are consistent across the 2x2 multiplication.
                        # The block index bi for A and B must be the SAME?
                        # No, the super-factor is a product of two linear combinations.
                        # L_A = sum_{ba in sf['a']} coeff_a[ba] * A_ba
                        # L_B = sum_{bb in sf['b']} coeff_b[bb] * B_bb
                        # M = L_A * L_B
                        # This M is a 2x2 matrix.
                        # The elements of M are sums of products of elements from A_ba and B_bb.
                        # M[li, lk] = sum_{ba, bb} coeff_a[ba] coeff_b[bb] * (A_ba * B_bb)[li, lk]
                        # (A_ba * B_bb)[li, lk] = sum_{lj} A_ba[li, lj] B_bb[lj, lk]
                        
                        # So we sum over ba, bb, and lj.
                        
                        # Let's restructure the loop.
                        
                        # We are inside the loop for r2.
                        # We need to add to U, V, W.
                        
                        # It is more efficient to precompute the non-zero (li, lj, lk) for each r2.
                        
                        # For now, let's just iterate.
                        
                        # We need to sum over ba in sf['a'] and bb in sf['b'].
                        
                        for ba, ca in sf['a'].items():
                            for bb, cb in sf['b'].items():
                                # The product term is ca * cb * cu * cv * cw
                                # But we must sum over lj?
                                # No, cu, cv, cw are for specific li, lj, lk.
                                # The term in the tensor is:
                                # U[global_i, global_j] += ca * cu
                                # V[global_j, global_k] += cb * cv
                                # W[global_k, global_i] += cw  (Wait, W index is k*n+i)
                                
                                # Wait, the Strassen decomposition is linear.
                                # The rank-1 tensor for the super-factor is:
                                # sum_{li,lj,lk} u2[li,r2] v2[lj,r2] w2[lk,r2] * (A_ba[li,lj] B_bb[lj,lk] C_ba/lk? No)
                                
                                # Let's stick to the definition:
                                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r]
                                
                                # For the super-factor, the contribution to T is:
                                # sum_{li,lj,lk} u2[li,r2] v2[lj,r2] w2[lk,r2] * 
                                #   (sum_{ba} ca[ba] A_ba[li,lj]) * (sum_{bb} cb[bb] B_bb[lj,lk]) * 
                                #   (delta for C? No, W is the output factor)
                                
                                # Actually, W is the factor for the output C.
                                # The identity is:
                                # C[i,k] = sum_j A[i,j] B[j,k]
                                # The decomposition is:
                                # C[i,k] = sum_r U[i,r] V[j,r] W[k,r]  (Sum over j? No, j is an index of V)
                                # Wait, V has shape (m*p, rank). Index is j*p+k.
                                # So V[j*p+k, r].
                                
                                # So the term is U[i, r] V[j*p+k, r] W[k*n+i, r].
                                
                                # In the block decomposition:
                                # i = 2*(bi_a//2) + li
                                # j = 2*(bi_a%2) + lj  (From A block)
                                # j must match the row of B.
                                # B block bb has rows 2*(bi_b//2) + lj.
                                # So we require 2*(bi_a%2) + lj == 2*(bi_b//2) + lj?
                                # This implies bi_a%2 == bi_b//2.
                                # This is not generally true for arbitrary ba, bb.
                                # However, in Strassen, the blocks are paired such that the dimensions match.
                                # A00 (rows 0-1, cols 0-1) multiplies B00 (rows 0-1, cols 0-1).
                                # So ba=0, bb=0.
                                # A00 (cols 0-1) matches B00 (rows 0-1).
                                # In general, A block ba has cols 2*(ba%2)..
                                # B block bb has rows 2*(bb//2)..
                                # For the product to be valid, the col range of A must match the row range of B.
                                # So 2*(ba%2) must equal 2*(bb//2).
                                # i.e., ba%2 == bb//2.
                                
                                # If this condition is not met, the term is 0.
                                
                                # Similarly for C.
                                # C block bc has rows 2*(bc//2).. and cols 2*(bc%2)..
                                # i (row of C) comes from A row: 2*(ba//2) + li.
                                # k (col of C) comes from B col: 2*(bb%2) + lk.
                                # So we require 2*(bc//2) + li == 2*(ba//2) + li => bc//2 == ba//2.
                                # And 2*(bc%2) + lk == 2*(bb%2) + lk => bc%2 == bb%2.
                                # So bc must be such that bc//2 == ba//2 and bc%2 == bb%2.
                                # This means bc is uniquely determined by ba and bb?
                                # No, ba and bb are summed over.
                                # But for a specific term in the sum, ba, bb, bc must be consistent.
                                
                                # In the Strassen super-factors, the combinations are structured such that this consistency holds.
                                # E.g., sf 0: A00+A11, B00+B11.
                                # Pairs: (0,0), (0,11)? No.
                                # (ba=0, bb=0): ba%2=0, bb//2=0. OK.
                                # (ba=0, bb=11): ba%2=0, bb//2=1. Mismatch.
                                # (ba=11, bb=0): ba%2=1, bb//2=0. Mismatch.
                                # (ba=11, bb=11): ba%2=1, bb//2=1. OK.
                                
                                # So only matching pairs contribute.
                                
                                # Let's add the terms.
                                
                                if (ba % 2) != (bb // 2):
                                    continue
                                    
                                # Determine bc
                                bc_row = ba // 2
                                bc_col = bb % 2
                                bc = bc_row * 2 + bc_col
                                
                                if bc not in sf['c']:
                                    continue
                                    
                                cc = sf['c'][bc]
                                
                                # Global indices
                                i_g = 2 * (ba // 2) + li
                                j_g = 2 * (ba % 2) + lj
                                k_g = 2 * (bb % 2) + lk
                                
                                # U index: i_g * m + j_g
                                u_idx = i_g * m + j_g
                                # V index: j_g * p + k_g
                                v_idx = j_g * p + k_g
                                # W index: k_g * n + i_g
                                w_idx = k_g * n + i_g
                                
                                # Coefficients
                                coeff_u = ca * cu
                                coeff_v = cb * cv
                                coeff_w = cc * cw
                                
                                U[u_idx][col_offset] += coeff_u
                                V[v_idx][col_offset] += coeff_v
                                W[w_idx][col_offset] += coeff_w
                                
            col_offset += 1
            
    # --- Part 2: Schoolbook 4x4x2 ---
    # Columns 4 and 5 of B.
    # Rank 32.
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col_offset] = 1
                V[v_idx][col_offset] = 1
                W[w_idx][col_offset] = 1
                col_offset += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
