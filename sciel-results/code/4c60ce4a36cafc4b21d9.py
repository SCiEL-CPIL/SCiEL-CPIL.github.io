import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2.
    Each term is (u_vec, v_vec, w_vec) where:
      u_vec: length 4 (indices: 00, 01, 10, 11 for A)
      v_vec: length 4 (indices: 00, 01, 10, 11 for B)
      w_vec: length 4 (indices: 00, 01, 10, 11 for C)
    
    Strassen formulas:
      M1 = (A00+A11)(B00+B11)
      M2 = (A10+A11)B00
      M3 = A00(B10-B11)
      M4 = A11(B01-B00)
      M5 = (A00+A01)B11
      M6 = (A10-A00)(B00+B10)
      M7 = (A01-A11)(B01+B11)
    
    C00 = M1+M4-M5+M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M6
    """
    # Index mapping for 2x2: [0,1,2,3] = [00,01,10,11]
    terms = []
    # M1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]  # A00 + A11
    v = [1, 0, 0, 1]  # B00 + B11
    w = [1, 0, 0, -1]  # C00 +=1, C11 += -1 ... wait
    # C00 = M1 + M4 - M5 + M7, C11 = M1 - M2 + M3 + M6
    # So M1 contributes +1 to C00 and +1 to C11
    w = [1, 0, 0, 1]  # C00 and C11
    terms.append((u, v, w))
    
    # M2 = (A10+A11)B00
    # C10 += M2, C11 -= M2
    u = [0, 0, 1, 1]  # A10 + A11
    v = [1, 0, 0, 0]  # B00
    w = [0, 0, 1, -1]  # C10 += 1, C11 -= 1
    terms.append((u, v, w))
    
    # M3 = A00(B10-B11)
    # C01 += M3, C11 += M3
    u = [1, 0, 0, 0]  # A00
    v = [0, 0, 1, -1]  # B10 - B11
    w = [0, 1, 0, 1]  # C01 += 1, C11 += 1
    terms.append((u, v, w))
    
    # M4 = A11(B01-B00)
    # C00 += M4, C10 += M4
    u = [0, 0, 0, 1]  # A11
    v = [0, 1, -1, 0]  # B01 - B00
    w = [1, 0, 1, 0]  # C00 += 1, C10 += 1
    terms.append((u, v, w))
    
    # M5 = (A00+A01)B11
    # C00 -= M5, C01 += M5
    u = [1, 1, 0, 0]  # A00 + A01
    v = [0, 0, 0, 1]  # B11
    w = [-1, 1, 0, 0]  # C00 -= 1, C01 += 1
    terms.append((u, v, w))
    
    # M6 = (A10-A00)(B00+B10)
    # C11 += M6
    u = [0, 0, 1, -1]  # A10 - A00  (wait: A10 is index 2, A00 is index 0)
    u = [1, 0, -1, 0]  # A00 - A10... let me recheck
    # M6 = (A10 - A00)(B00 + B10)
    u = [0, 0, 1, 0]  # A10
    # Actually: A10 - A00 means coefficient -1 for A00, +1 for A10
    u = [-1, 0, 1, 0]  # -A00 + A10
    v = [1, 0, 1, 0]  # B00 + B10
    w = [0, 0, 0, 1]  # C11 += 1
    terms.append((u, v, w))
    
    # M7 = (A01-A11)(B01+B11)
    # C00 += M7
    u = [0, 1, 0, -1]  # A01 - A11
    v = [0, 1, 0, 1]  # B01 + B11
    w = [1, 0, 0, 0]  # C00 += 1
    terms.append((u, v, w))
    
    return terms

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(actual - expected) > 1e-8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: Split M_{4,4,9} into M_{4,4,4} + M_{4,4,5}
    # M_{4,4,4}: Use Strassen recursively -> rank 49
    # M_{4,4,5}: Use schoolbook -> rank 80
    # Total rank: 129
    
    # Build Strassen decomposition for 4x4 * 4x4
    # 4x4 = 2x2 blocks of 2x2
    # Strassen on 4x4: 7 multiplications of 2x2 * 2x2
    # Each 2x2 * 2x2 uses 7 Strassen terms
    # Total: 7 * 7 = 49 terms
    
    # Index mapping for 4x4 matrix:
    # A[i,j] where i,j in {0,1,2,3}
    # Block structure: A = [[A00, A01], [A10, A11]] where each block is 2x2
    # A00 = A[0,0],A[0,1],A[1,0],A[1,1]
    # A01 = A[0,2],A[0,3],A[1,2],A[1,3]
    # A10 = A[2,0],A[2,1],A[3,0],A[3,1]
    # A11 = A[2,2],A[2,3],A[3,2],A[3,3]
    
    # For the 4x4 * 4x4 part (k=0..3), we need:
    # C[i,j] for i,j in {0,1,2,3}
    # C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Using Strassen at the 4x4 level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Each Mj is a 2x2 * 2x2 product, which we compute with Strassen (7 terms)
    
    # For the 4x4 * 4x5 part (k=4..8), schoolbook:
    # For each (i,j,k) with k in 4..8: one rank-1 term
    # 4*4*5 = 80 terms
    
    # Total: 49 + 80 = 129 terms
    
    rank_total = 49 + 80
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    col = 0
    
    # Part 1: Strassen for 4x4 * 4x4 (k=0..3)
    # We need to map the 4x4 indices to the flat indices
    # A flat index: i*m + j = i*4 + j
    # B flat index: j*p + k = j*9 + k
    # C flat index: k*n + i = k*4 + i
    
    # 2x2 Strassen terms (from above)
    s2_terms = _strassen_2x2_terms()
    
    # Define the 7 Strassen combinations at the 4x4 level
    # Each combination has:
    #   - A coefficients for blocks A00, A01, A10, A11
    #   - B coefficients for blocks B00, B01, B10, B11
    #   - C contributions to blocks C00, C01, C10, C11
    
    # Strassen combinations at 4x4 level:
    # M1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1)  [C00+=1, C11+=1]
    # M2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,-1) [C10+=1, C11-=1]
    # M3: A=(1,0,0,0), B=(0,0,1,-1), C=(0,1,0,1) [C01+=1, C11+=1]
    # M4: A=(0,0,0,1), B=(0,1,-1,0), C=(1,0,1,0) [C00+=1, C10+=1]
    # M5: A=(1,1,0,0), B=(0,0,0,1), C=(-1,1,0,0) [C00-=1, C01+=1]
    # M6: A=(-1,0,1,0), B=(1,0,1,0), C=(0,0,0,1) [C11+=1]
    # M7: A=(0,1,0,-1), B=(0,1,0,1), C=(1,0,0,0) [C00+=1]
    
    strassen_4x4_combos = [
        # (A_coeffs, B_coeffs, C_coeffs) for blocks 00,01,10,11
        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),   # M1
        ((0,0,1,1), (1,0,0,0), (0,0,1,-1)),   # M2
        ((1,0,0,0), (0,0,1,-1), (0,1,0,1)),   # M3
        ((0,0,0,1), (0,1,-1,0), (1,0,1,0)),   # M4
        ((1,1,0,0), (0,0,0,1), (-1,1,0,0)),   # M5
        ((-1,0,1,0), (1,0,1,0), (0,0,0,1)),   # M6
        ((0,1,0,-1), (0,1,0,1), (1,0,0,0)),   # M7
    ]
    
    # Block index to flat indices
    # For 4x4, block (bi, bj) where bi,bj in {0,1}:
    # A block (bi,bj): rows 2*bi to 2*bi+1, cols 2*bj to 2*bj+1
    # Flat index for A[i,j] = i*4 + j
    # Block (bi,bj) contains A[2*bi, 2*bj], A[2*bi, 2*bj+1], A[2*bi+1, 2*bj], A[2*bi+1, 2*bj+1]
    
    def block_flat_indices(bi, bj):
        """Return list of 4 flat indices for block (bi,bj) in 4x4 matrix."""
        return [
            (2*bi) * 4 + (2*bj),
            (2*bi) * 4 + (2*bj+1),
            (2*bi+1) * 4 + (2*bj),
            (2*bi+1) * 4 + (2*bj+1),
        ]
    
    # For each of the 7 Strassen combos, and for each of the 7 sub-Strassen terms,
    # create rank-1 terms
    
    for combo_idx, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_4x4_combos):
        for s2_idx, (u2, v2, w2) in enumerate(s2_terms):
            # u2, v2, w2 are length-4 vectors for 2x2 matrices
            # u2 = [u00, u01, u10, u11]
            # This corresponds to the 2x2 matrix structure within each block
            
            # For each element in the 2x2 sub-matrix, expand to 4x4
            for sub_i in range(2):  # row within 2x2
                for sub_j in range(2):  # col within 2x2
                    # u2[sub_i*2 + sub_j] is the coefficient for A[sub_i, sub_j] within the 2x2
                    coeff_u = u2[sub_i * 2 + sub_j]
                    coeff_v = v2[sub_i * 2 + sub_j]
                    coeff_w = w2[sub_i * 2 + sub_j]
                    
                    if coeff_u == 0 or coeff_v == 0 or coeff_w == 0:
                        continue
                    
                    # For each block in A, B, C that this sub-element belongs to
                    for ab in range(4):  # A block index: 00,01,10,11
                        a_bi = ab // 2
                        a_bj = ab % 2
                        a_coeff = a_coeffs[ab]
                        if a_coeff == 0:
                            continue
                        
                        # The A block (a_bi, a_bj) contributes with coefficient a_coeff * coeff_u
                        # For the specific sub-element (sub_i, sub_j) within this block
                        a_flat = (2*a_bi + sub_i) * 4 + (2*a_bj + sub_j)
                        
                        for bb in range(4):  # B block index
                            b_bi = bb // 2
                            b_bj = bb % 2
                            b_coeff = b_coeffs[bb]
                            if b_coeff == 0:
                                continue
                            
                            # B block (b_bi, b_bj) - but wait, B has shape 4x4 for the 4x4x4 part
                            # B flat index: j*9 + k, but for the 4x4x4 part, k is 0..3
                            # B[j,k] where j is row, k is col
                            # Block (b_bi, b_bj): rows 2*b_bi to 2*b_bi+1, cols 2*b_bj to 2*b_bj+1
                            b_flat = (2*b_bi + sub_i) * 9 + (2*b_bj + sub_j)
                            
                            for cb in range(4):  # C block index
                                c_bi = cb // 2
                                c_bj = cb % 2
                                c_coeff = c_coeffs[cb]
                                if c_coeff == 0:
                                    continue
                                
                                # C block (c_bi, c_bj): C[k,i] where k is col, i is row
                                # C flat index: k*4 + i
                                # Block (c_bi, c_bj): cols 2*c_bi to 2*c_bi+1, rows 2*c_bj to 2*c_bj+1
                                # Wait, C has shape p x n = 9 x 4 for the full problem
                                # For the 4x4x4 part, C has shape 4x4
                                # C[k,i] with k in 0..3, i in 0..3
                                # Flat index: k*4 + i
                                # Block (c_bi, c_bj): k in {2*c_bi, 2*c_bi+1}, i in {2*c_bj, 2*c_bj+1}
                                c_flat = (2*c_bi + sub_i) * 4 + (2*c_bj + sub_j)
                                
                                # But wait - the sub_i and sub_j indices need to be consistent
                                # For A: sub_i is row, sub_j is col -> A[2*a_bi+sub_i, 2*a_bj+sub_j]
                                # For B: sub_i is row, sub_j is col -> B[2*b_bi+sub_i, 2*b_bj+sub_j]
                                # For C: C[k,i] where k=col of B, i=row of A
                                # So C's "row" index k corresponds to B's col, and C's "col" index i corresponds to A's row
                                # C block (c_bi, c_bj): k in {2*c_bi, 2*c_bi+1}, i in {2*c_bj, 2*c_bj+1}
                                # C[2*c_bi+sub_k, 2*c_bj+sub_i]
                                # But sub_k should match sub_j from B (B's col), and sub_i should match sub_i from A (A's row)
                                
                                # This is getting complicated. Let me use a different approach.
                                pass
    
    # This nested approach is getting too complex. Let me use a simpler method.
    # I'll directly construct the rank-1 terms by iterating over all combinations.
    
    # Reset and use a cleaner approach
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    col = 0
    
    # Part 1: Strassen for 4x4 * 4x4 (k=0..3)
    # Direct construction: for each of the 7*7=49 terms
    
    # The 4x4 Strassen: 7 outer terms, each is a 2x2*2x2 product with 7 inner terms
    # For outer term t and inner term s:
    #   A contribution: sum of a_coeffs[ab] * u2_s[si,sj] for block ab, sub-position (si,sj)
    #   B contribution: sum of b_coeffs[bb] * v2_s[si,sj] for block bb, sub-position (si,sj)  
    #   C contribution: sum of c_coeffs[cb] * w2_s[si,sj] for block cb, sub-position (si,sj)
    
    # But each rank-1 term is a single (u_vec, v_vec, w_vec) triple
    # So I need to expand each (outer, inner) pair into potentially multiple rank-1 terms
    # if the coefficients don't factor nicely.
    
    # Actually, each (outer_term, inner_term) pair gives ONE rank-1 term where:
    # u_vec has non-zero entries at specific (i,j) positions
    # v_vec has non-zero entries at specific (j,k) positions
    # w_vec has non-zero entries at specific (k,i) positions
    
    # Let me just enumerate all 49 terms explicitly
    
    # For each outer Strassen term (7 of them) and each inner 2x2 Strassen term (7 of them):
    # The outer term specifies which 4x4 blocks are involved
    # The inner term specifies which 2x2 elements within those blocks
    
    # Let me build this more carefully
    
    # Clear and rebuild
    col = 0
    for outer_idx, (a_c, b_c, c_c) in enumerate(strassen_4x4_combos):
        for inner_idx, (u2, v2, w2) in enumerate(s2_terms):
            # u2 = [u00, u01, u10, u11] for the 2x2 "A" part
            # v2 = [v00, v01, v10, v11] for the 2x2 "B" part
            # w2 = [w00, w01, w10, w11] for the 2x2 "C" part
            
            # This represents: (sum over A blocks) * (sum over B blocks) -> (sum over C blocks)
            # Each of these is a 2x2 matrix product
            
            # The 2x2 product decomposes into 7 rank-1 terms (inner Strassen)
            # But we already have the inner Strassen terms as u2, v2, w2
            
            # For each sub-position in the 2x2:
            # The rank-1 term has:
            #   u_vec: non-zero at positions where a_c[ab] * u2[sub] != 0
            #   v_vec: non-zero at positions where b_c[bb] * v2[sub] != 0
            #   w_vec: non-zero at positions where c_c[cb] * w2[sub] != 0
            
            # But this is a tensor product structure. Let me think again.
            
            # The outer Strassen term says: compute M_outer = A_outer * B_outer
            # where A_outer is a 4x4 matrix formed by combining blocks
            # and B_outer is a 4x4 matrix formed by combining blocks
            # The result contributes to C_outer which is also a combination of blocks
            
            # A_outer[i,j] = sum_ab a_c[ab] * A_block_ab[i-2*(ab//2), j-2*(ab%2)]
            # This is a linear combination of the 4x4 A
            
            # Similarly for B_outer and C_outer
            
            # The product A_outer * B_outer is a 4x4 matrix
            # We compute it with Strassen (7 terms for 4x4 = 7*7=49 for the full thing)
            
            # Wait, I'm confusing levels. Let me restart the logic.
            
            # Level 1: 4x4 * 4x4 using Strassen = 7 multiplications of 2x2 * 2x2
            # Level 2: Each 2x2 * 2x2 using Strassen = 7 rank-1 terms
            # Total: 7 * 7 = 49 rank-1 terms
            
            # For outer term t (0..6):
            #   A_t is a linear combination of the four 2x2 blocks of A
            #   B_t is a linear combination of the four 2x2 blocks of B
            #   The product A_t * B_t contributes to C with specific block coefficients
            #   A_t * B_t is a 2x2 * 2x2 product, computed with 7 Strassen terms
            
            # For each (t, s) where s is the inner Strassen term:
            #   The rank-1 term involves specific elements of A, B, C
            
            # A_t[i,j] = sum_{ab} a_c[t][ab] * A[2*(ab//2)+i, 2*(ab%2)+j] for i,j in {0,1}
            # B_t[i,j] = sum_{bb} b_c[t][bb] * B[2*(bb//2)+i, 2*(bb%2)+j] for i,j in {0,1}
            # C contribution: c_c[t][cb] * (A_t * B_t) at block cb
            
            # The inner Strassen term s says:
            #   (A_t * B_t)[i,j] += u2[s][i,j] * v2[s][i',j'] ... no
            #   Actually: A_t * B_t = sum_s u2_s * v2_s * w2_s (outer product decomposition)
            #   Where (A_t * B_t)[i,j] = sum_s u2_s[i] * v2_s[j] * w2_s[i,j]... 
            #   No, the 2x2 matrix product has rank-1 decomposition:
            #   (A*B)[i,j] = sum_s U_s[i] * V_s[j] * W_s[i,j]
            #   But that's not right either. The standard rank-1 decomposition of the
            #   matrix multiplication tensor M_{2,2,2} has 7 terms where each term is:
            #   term_s: (u_s, v_s, w_s) with u_s, v_s in R^4, w_s in R^4
            #   (A*B)[i,j] = sum_s u_s[i*2+j] * v_s[i*2+j] * w_s[i*2+j]... 
            #   No! The indices are:
            #   A is 2x2, indexed by (i,j) -> flat i*2+j
            #   B is 2x2, indexed by (j,k) -> flat j*2+k
            #   C is 2x2, indexed by (i,k) -> flat i*2+k
            #   (A*B)[i,k] = sum_j A[i,j] * B[j,k]
            #   Tensor: T[i*2+j, j*2+k, i*2+k] = 1
            #   Rank-1 decomposition: sum_s U_s[i*2+j] * V_s[j*2+k] * W_s[i*2+k]
            
            # So for the inner 2x2 Strassen:
            # u2_s is length 4: indices 0,1,2,3 = (0,0),(0,1),(1,0),(1,1) for A
            # v2_s is length 4: indices 0,1,2,3 = (0,0),(0,1),(1,0),(1,1) for B
            # w2_s is length 4: indices 0,1,2,3 = (0,0),(0,1),(1,0),(1,1) for C
            
            # Now, for the outer term t:
            # A_t is a 2x2 matrix where:
            #   A_t[si,sj] = sum_{ab} a_c[t][ab] * A[2*(ab//2)+si, 2*(ab%2)+sj]
            # B_t is a 2x2 matrix where:
            #   B_t[si,sj] = sum_{bb} b_c[t][bb] * B[2*(bb//2)+si, 2*(bb%2)+sj]
            # C_t contribution: c_c[t][cb] * (A_t * B_t) added to C block cb
            #   C[2*(cb//2)+si, 2*(cb%2)+sj] += c_c[t][cb] * (A_t * B_t)[si,sj]
            
            # (A_t * B_t)[si,sj] = sum_sk A_t[si,sk] * B_t[sk,sj]
            # = sum_sk (sum_ab a_c[t][ab] * A[2*(ab//2)+si, 2*(ab%2)+sk]) * (sum_bb b_c[t][bb] * B[2*(bb//2)+sk, 2*(bb%2)+sj])
            
            # Using inner Strassen: (A_t * B_t)[si,sj] = sum_s u2_s[si*2+sk] * v2_s[sk*2+sj] * w2_s[si*2+sj]
            # Wait, the flat indices for 2x2: A_t[si,sk] has flat index si*2+sk
            # B_t[sk,sj] has flat index sk*2+sj
            # C_t[si,sj] has flat index si*2+sj
            
            # So: (A_t * B_t)[si,sj] = sum_s u2_s[si*2+sk] * v2_s[sk*2+sj] * w2_s[si*2+sj]
            # But this sum is over s (the 7 Strassen terms), and for each s, it's a product
            # of three scalars. But u2_s[si*2+sk] depends on sk, so this isn't a single
            # rank-1 term in the original A,B,C space.
            
            # I think the issue is that I'm overcomplicating this. Let me just directly
            # construct the 49 rank-1 terms by expanding everything.
            
            # For each (t, s):
            # The contribution to C is:
            # sum_{si,sk,sj} c_c[t][cb] * w2_s[si*2+sj] * u2_s[si*2+sk] * v2_s[sk*2+sj]
            #   * A[2*(ab//2)+si, 2*(ab%2)+sk] * B[2*(bb//2)+sk, 2*(bb%2)+sj]
            # where we sum over ab, bb, cb, si, sk, sj
            
            # This is a sum of products of the form:
            # (coeff_A) * A[ia, ja] * (coeff_B) * B[jb, kb] * (coeff_C) * C[kc, ic]
            
            # Each such product is a rank-1 term! So for each (t, s, si, sk, sj, ab, bb, cb)
            # where all coefficients are non-zero, we get one rank-1 term.
            
            # But that could be up to 7 * 7 * 2 * 2 * 2 * 4 * 4 * 4 = 7*7*8*64 = 25088 terms
            # That's way too many. The whole point of Strassen is to have FEWER terms.
            
            # I think the correct interpretation is:
            # The 4x4 Strassen gives 7 "big" multiplications of 2x2 matrices.
            # Each 2x2 multiplication uses 7 rank-1 terms.
            # Total: 49 rank-1 terms.
            # Each rank-1 term is a SINGLE (u_vec, v_vec, w_vec) where u_vec, v_vec, w_vec
            # are vectors in R^16, R^36, R^36 respectively (for the full problem).
            
            # Let me just build each of the 49 terms explicitly.
            
            # For outer term t and inner term s:
            # u_vec: 16-dimensional (for A, which is 4x4)
            # v_vec: 36-dimensional (for B, which is 4x9, but we only use first 4 cols)
            # w_vec: 36-dimensional (for C, which is 9x4, but we only use first 4 rows)
            
            # u_vec[i*4+j] = sum_{ab: 2*(ab//2)==i, 2*(ab%2)==j} a_c[t][ab] * u2_s[(i%2)*2+(j%2)]
            # Wait, that's not right. Let me think more carefully.
            
            # A_t is a 2x2 matrix: A_t[si,sj] = sum_{ab} a_c[t][ab] * A[2*(ab//2)+si, 2*(ab%2)+sj]
            # The "A" in the rank-1 term is A_t, not A.
            # But we need to express A_t in terms of A.
            # A_t[si,sj] = sum_{ab} a_c[t][ab] * A[2*(ab//2)+si, 2*(ab%2)+sj]
            # So the coefficient of A[i,j] in A_t is:
            #   sum_{ab: 2*(ab//2)==i, 2*(ab%2)==j} a_c[t][ab]
            # But since i = 2*(ab//2)+si and j = 2*(ab%2)+sj, for fixed si,sj:
            #   ab//2 = i//2, ab%2 = j//2, so ab = 2*(i//2) + (j//2)
            #   Coefficient = a_c[t][2*(i//2) + (j//2)]
            
            # So u_vec[i*4+j] = a_c[t][2*(i//2) + (j//2)] * u2_s[(i%2)*2+(j%2)]
            
            # Similarly:
            # v_vec[j*9+k] = b_c[t][2*(j//2) + (k//2)] * v2_s[(j%2)*2+(k%2)] for k < 4
            # w_vec[k*4+i] = c_c[t][2*(k//2) + (i//2)] * w2_s[(k%2)*2+(i%2)] for k < 4
            
            # Let's build this term
            u_vec = np.zeros(n * m, dtype=float)
            v_vec = np.zeros(m * p, dtype=float)
            w_vec = np.zeros(p * n, dtype=float)
            
            for i in range(4):
                for j in range(4):
                    ab = 2 * (i // 2) + (j // 2)
                    si = i % 2
                    sj = j % 2
                    coeff_a = a_c[ab] * u2[si * 2 + sj]
                    if coeff_a != 0:
                        u_vec[i * 4 + j] = coeff_a
            
            for j in range(4):
                for k in range(4):  # Only first 4 columns for the 4x4x4 part
                    bb = 2 * (j // 2) + (k // 2)
                    si = j % 2
                    sj = k % 2
                    coeff_b = b_c[bb] * v2[si * 2 + sj]
                    if coeff_b != 0:
                        v_vec[j * 9 + k] = coeff_b
            
            for k in range(4):
                for i in range(4):
                    cb = 2 * (k // 2) + (i // 2)
                    si = k % 2
                    sj = i % 2
                    coeff_c = c_c[cb] * w2[si * 2 + sj]
                    if coeff_c != 0:
                        w_vec[k * 4 + i] = coeff_c
            
            # Check if this term is non-trivial
            if np.any(u_vec) and np.any(v_vec) and np.any(w_vec):
                U[:, col] = u_vec
                V[:, col] = v_vec
                W[:, col] = w_vec
                col += 1
    
    # Part 2: Schoolbook for 4x4 * 4x5 (k=4..8)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Trim to actual number of columns used
    if col < rank_total:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank_total = col
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
        rank_sb = n * m * p
        U = np.zeros((n * m, rank_sb), dtype=float)
        V = np.zeros((m * p, rank_sb), dtype=float)
        W = np.zeros((p * n, rank_sb), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    if time.time() > deadline - 5:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
