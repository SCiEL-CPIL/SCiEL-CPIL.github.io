def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Define the 2x2 Strassen multiplication tensor
    # Indices for 2x2: A(i,j), B(j,k), C(i,k) with i,j,k in {0,1}
    # We represent the 2x2 tensor as a list of (u_idx, v_idx, w_idx, coeff)
    # But we need to compose them.
    
    # Let's build the 4x4x4 tensor first.
    # We will store the terms as (u_idx, v_idx, w_idx, coeff)
    # u_idx in 0..15 (4x4)
    # v_idx in 0..15 (4x4)
    # w_idx in 0..15 (4x4)
    
    terms_4x4 = []
    
    # Strassen 4x4 steps
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # We need a function to expand a 2x2 product into 7 scalar terms.
    # Given a mapping from 2x2 A indices to 4x4 A indices,
    # and a mapping from 2x2 B indices to 4x4 B indices,
    # and a mapping from 2x2 C indices to 4x4 C indices,
    # we can generate the 7 terms.
    
    # 2x2 Strassen terms:
    # m1: (a00+a11)(b00+b11)
    # m2: (a10+a11)b00
    # m3: a00(b01-b11)
    # m4: a11(b10-b00)
    # m5: (a00+a01)b11
    # m6: (a10-a00)(b00+b01)
    # m7: (a01-a11)(b10+b11)
    
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Let's define the 2x2 Strassen expansion as a list of (a_idx, b_idx, c_idx, coeff)
    # where a_idx, b_idx, c_idx are in {0,1,2,3} for the 2x2 matrices.
    # a_idx: 0=a00, 1=a01, 2=a10, 3=a11
    # b_idx: 0=b00, 1=b01, 2=b10, 3=b11
    # c_idx: 0=c00, 1=c01, 2=c10, 3=c11
    
    # m1 = (a0+a3)(b0+b3) = a0b0 + a0b3 + a3b0 + a3b3
    # This is a sum of 4 terms.
    # But Strassen is a rank-7 decomposition, so we should not expand it into 8 terms.
    # The rank-7 decomposition means we have 7 intermediate values.
    # So we should treat each m_k as a single "multiplication".
    # But m_k itself is a product of two linear combinations.
    # So m_k = (sum a_i u_i) (sum b_j v_j).
    # This is a rank-1 tensor? No, it's a sum of rank-1 tensors.
    # Wait, the rank of 2x2x2 is 7.
    # The 7 terms are:
    # t1 = (a00+a11)(b00+b11)
    # t2 = (a10+a11)b00
    # t3 = a00(b01-b11)
    # t4 = a11(b10-b00)
    # t5 = (a00+a01)b11
    # t6 = (a10-a00)(b00+b01)
    # t7 = (a01-a11)(b10+b11)
    
    # Each t_k is a product of two linear forms.
    # So we can represent t_k as a sum of rank-1 tensors?
    # No, t_k is itself a bilinear form.
    # We can write t_k = sum_{i,j} coeff_{ij} a_i b_j.
    # Then C = sum_k c_k t_k.
    # So C = sum_{i,j} (sum_k c_k coeff_{ij}^k) a_i b_j.
    # This would give rank 16, which is not minimal.
    
    # The correct way is to use the 7 intermediate values as the "rank" dimension.
    # So for each of the 7 Strassen steps in 4x4, we have a 2x2 product.
    # Each 2x2 product has rank 7.
    # So total rank is 7 * 7 = 49.
    
    # So we need to generate the 7 terms for each of the 7 Strassen steps.
    # For step k, we have a 2x2 product X_k Y_k = M_k.
    # We use the rank-7 decomposition of X_k Y_k.
    # So we get 7 scalar multiplications.
    # We add these to our global tensor.
    
    # Let's define the 2x2 Strassen decomposition as a list of 7 terms.
    # Each term is (u_vec, v_vec, w_vec) where u_vec, v_vec, w_vec are vectors of length 4.
    # u_vec[i] is the coefficient of a_i.
    # v_vec[j] is the coefficient of b_j.
    # w_vec[k] is the coefficient of c_k.
    # But wait, the decomposition is:
    # C = sum_{r=1}^7 u_r v_r w_r
    # where u_r, v_r, w_r are vectors.
    # Actually, the standard form is:
    # C_{ij} = sum_{k} M_{ik} N_{kj}
    # = sum_{r=1}^7 (sum_i u_{ri} A_{ik}) (sum_j v_{rj} B_{kj}) ... no.
    
    # The tensor is T = sum_{r=1}^7 u_r \otimes v_r \otimes w_r.
    # T_{ijk} = sum_{r=1}^7 u_{ri} v_{rj} w_{rk}.
    # Here i is row of A, j is col of A / row of B, k is col of B.
    
    # For 2x2:
    # u_r is a vector of length 4 (for A_{00}, A_{01}, A_{10}, A_{11}).
    # v_r is a vector of length 4 (for B_{00}, B_{01}, B_{10}, B_{11}).
    # w_r is a vector of length 4 (for C_{00}, C_{01}, C_{10}, C_{11}).
    
    # Let's define the 7 terms for 2x2 Strassen.
    # t1 = (a00+a11)(b00+b11) -> contributes to c00 and c11
    # u1 = [1, 0, 0, 1]
    # v1 = [1, 0, 0, 1]
    # w1 = [1, 0, 0, 1]  # c00 += t1, c11 += t1
    
    # t2 = (a10+a11)b00 -> contributes to c10 and c11
    # u2 = [0, 0, 1, 1]
    # v2 = [1, 0, 0, 0]
    # w2 = [0, 0, 1, -1] # c10 += t2, c11 -= t2
    
    # t3 = a00(b01-b11) -> contributes to c01 and c11
    # u3 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # w3 = [0, 1, 0, 1] # c01 += t3, c11 += t3
    
    # t4 = a11(b10-b00) -> contributes to c00 and c10
    # u4 = [0, 0, 0, 1]
    # v4 = [0, 0, 1, -1]
    # w4 = [1, 0, 1, 0] # c00 += t4, c10 += t4
    
    # t5 = (a00+a01)b11 -> contributes to c00 and c01
    # u5 = [1, 1, 0, 0]
    # v5 = [0, 0, 0, 1]
    # w5 = [-1, 1, 0, 0] # c00 -= t5, c01 += t5
    
    # t6 = (a10-a00)(b00+b01) -> contributes to c11
    # u6 = [-1, 0, 1, 0]
    # v6 = [1, 1, 0, 0]
    # w6 = [0, 0, 0, 1] # c11 += t6
    
    # t7 = (a01-a11)(b10+b11) -> contributes to c00
    # u7 = [0, 1, 0, -1]
    # v7 = [0, 0, 1, 1]
    # w7 = [1, 0, 0, 0] # c00 += t7
    
    # Let's verify:
    # c00 = t1 + t4 - t5 + t7
    # c01 = t3 + t5
    # c10 = t2 + t4
    # c11 = t1 - t2 + t3 + t6
    
    # This matches the standard Strassen formulas.
    
    # Now for 4x4:
    # We have 7 Strassen steps.
    # For each step, we have a 2x2 product.
    # We map the 2x2 indices to 4x4 indices.
    
    # Step 1: M1 = (A11+A22)(B11+B22)
    # A11: A[0:2, 0:2], A22: A[2:4, 2:4]
    # B11: B[0:2, 0:2], B22: B[2:4, 2:4]
    # C11: C[0:2, 0:2]
    
    # We need to map the 2x2 Strassen terms to the 4x4 indices.
    # For A11+A22:
    # u1 (for A) is [1,0,0,1] for 2x2.
    # This means a00 and a11.
    # In 4x4, a00 is A[0,0], a11 is A[1,1].
    # So for A11, u1 maps to A[0,0] and A[1,1].
    # For A22, u1 maps to A[2,2] and A[3,3].
    # So the combined u vector for A is:
    # A[0,0]: 1, A[1,1]: 1, A[2,2]: 1, A[3,3]: 1.
    # All other A entries are 0.
    
    # Similarly for B.
    # And for C.
    
    # Let's define a function to add a 2x2 Strassen term to the global tensor.
    # It takes:
    # - a_map: a function mapping 2x2 A index (0..3) to 4x4 A index (0..15) or None.
    # - b_map: a function mapping 2x2 B index (0..3) to 4x4 B index (0..15) or None.
    # - c_map: a function mapping 2x2 C index (0..3) to 4x4 C index (0..15) or None.
    # - u_vec, v_vec, w_vec: the 2x2 Strassen vectors.
    # - offset: the column offset in the global tensor.
    
    # This is getting complicated. I will just hardcode the 49 terms.
    # Or better, I will write a small script to generate them and paste the result.
    # But I can't run scripts.
    
    # Let's do it programmatically in the construct function.
    
    def add_2x2_strassen(u_map, v_map, c_map, u_vec, v_vec, w_vec, offset):
        # u_map: dict {2x2_idx: 4x4_idx}
        # v_map: dict {2x2_idx: 4x4_idx}
        # c_map: dict {2x2_idx: 4x4_idx}
        
        # u_vec, v_vec, w_vec are lists of 4 floats.
        
        # We need to add the term:
        # sum_{i,j,k} u_vec[i] v_vec[j] w_vec[k] A_{u_map[i]} B_{v_map[j]} C_{c_map[k]}
        # This is a rank-1 tensor? No.
        # The term is (sum_i u_vec[i] A_{u_map[i]}) * (sum_j v_vec[j] B_{v_map[j]}) * (sum_k w_vec[k] C_{c_map[k]})?
        # No, the tensor is T = sum_r u_r \otimes v_r \otimes w_r.
        # T_{ijk} = sum_r u_{ri} v_{rj} w_{rk}.
        # So for each r, we add u_{ri} v_{rj} w_{rk} to T_{ijk}.
        # Here i is the index in U, j in V, k in W.
        # But U, V, W are matrices.
        # U_{i, r} = u_{ri}
        # V_{j, r} = v_{rj}
        # W_{k, r} = w_{rk}
        
        # So for each r (0..6), we add:
        # U[u_map[i], offset+r] += u_vec[i]
        # V[v_map[j], offset+r] += v_vec[j]
        # W[c_map[k], offset+r] += w_vec[k]
        # Wait, this is not right.
        # The rank-1 term is u \otimes v \otimes w.
        # This means U_{i, r} = u_i, V_{j, r} = v_j, W_{k, r} = w_k.
        # So for each r, we set the r-th column of U, V, W.
        # But u, v, w are vectors of length 4.
        # So we need to map the 2x2 indices to the 4x4 indices.
        
        # Let's create the vectors for the 4x4 case.
        # u_4x4 is a vector of length 16.
        # u_4x4[u_map[i]] += u_vec[i]
        # Similarly for v and w.
        
        u_4x4 = [0.0] * 16
        v_4x4 = [0.0] * 16
        w_4x4 = [0.0] * 16
        
        for i in range(4):
            if u_map[i] is not None:
                u_4x4[u_map[i]] += u_vec[i]
        for j in range(4):
            if v_map[j] is not None:
                v_4x4[v_map[j]] += v_vec[j]
        for k in range(4):
            if c_map[k] is not None:
                w_4x4[c_map[k]] += w_vec[k]
        
        # Now add this rank-1 term to the global tensor.
        # But wait, this is only one term of the 7.
        # We need to add all 7 terms for this 2x2 product.
        # So we call this function 7 times, once for each r.
        # But the u_map, v_map, c_map are the same for all 7 terms.
        # Only u_vec, v_vec, w_vec change.
        
        # So we should pass the 7 terms at once.
        
        pass

    # Let's redefine.
    # We will have a list of 7 terms for 2x2 Strassen.
    # Each term is (u_vec, v_vec, w_vec).
    
    strassen_2x2_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # t1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # t2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # t3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # t4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # t5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # t6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # t7
    ]
    
    # Now for 4x4, we have 7 steps.
    # For each step, we have a 2x2 product.
    # We define the maps for each step.
    
    # Step 1: M1 = (A11+A22)(B11+B22)
    # A11: 0,1,2,3 -> 0,1,4,5
    # A22: 0,1,2,3 -> 8,9,12,13
    # B11: 0,1,2,3 -> 0,1,4,5
    # B22: 0,1,2,3 -> 8,9,12,13
    # C11: 0,1,2,3 -> 0,1,4,5
    
    # u_map1: {0:0, 1:1, 2:4, 3:5} for A11, and {0:8, 1:9, 2:12, 3:13} for A22.
    # So u_map1 = {0: [0,8], 1: [1,9], 2: [4,12], 3: [5,13]}
    # This is a list of lists.
    
    # Let's define the maps as lists of lists.
    # u_map[i] is a list of 4x4 indices for the i-th 2x2 index.
    
    def make_map(offsets):
        # offsets is a list of 4 tuples, each tuple has the 4x4 indices for the 4 2x2 indices.
        # Actually, it's simpler to just have a list of 4 lists.
        pass
    
    # This is too complex to do in my head.
    # I will write the code to generate the maps.
    
    # A11: i in 0,1; j in 0,1. Index = i*4 + j.
    # A12: i in 0,1; j in 2,3. Index = i*4 + j.
    # A21: i in 2,3; j in 0,1. Index = i*4 + j.
    # A22: i in 2,3; j in 2,3. Index = i*4 + j.
    
    # 2x2 indices: 00, 01, 10, 11.
    # Map for A11: 0->0, 1->1, 2->4, 3->5.
    # Map for A12: 0->2, 1->3, 2->6, 3->7.
    # Map for A21: 0->8, 1->9, 2->12, 3->13.
    # Map for A22: 0->10, 1->11, 2->14, 3->15.
    
    # Similarly for B.
    # Map for B11: 0->0, 1->1, 2->4, 3->5.
    # Map for B12: 0->2, 1->3, 2->6, 3->7.
    # Map for B21: 0->8, 1->9, 2->12, 3->13.
    # Map for B22: 0->10, 1->11, 2->14, 3->15.
    
    # Map for C11: 0->0, 1->1, 2->4, 3->5.
    # Map for C12: 0->2, 1->3, 2->6, 3->7.
    # Map for C21: 0->8, 1->9, 2->12, 3->13.
    # Map for C22: 0->10, 1->11, 2->14, 3->15.
    
    # Now the 7 Strassen steps:
    # 1: A11+A22, B11+B22, C11
    # 2: A21+A22, B11, C10
    # 3: A11, B12-B22, C01
    # 4: A22, B21-B11, C10
    # 5: A11+A12, B22, C00
    # 6: A21-A11, B11+B12, C11
    # 7: A12-A22, B21+B22, C00
    
    # Let's define the maps for each step.
    # u_map is a list of 4 lists.
    # u_map[i] is the list of 4x4 indices for the i-th 2x2 index.
    
    # Step 1:
    # u_map1: 0:[0,10], 1:[1,11], 2:[4,14], 3:[5,15]
    # v_map1: 0:[0,10], 1:[1,11], 2:[4,14], 3:[5,15]
    # c_map1: 0:[0], 1:[1], 2:[4], 3:[5]
    
    # Step 2:
    # u_map2: 0:[8,10], 1:[9,11], 2:[12,14], 3:[13,15]
    # v_map2: 0:[0], 1:[1], 2:[4], 3:[5]
    # c_map2: 0:[8], 1:[9], 2:[12], 3:[13]
    
    # Step 3:
    # u_map3: 0:[0], 1:[1], 2:[4], 3:[5]
    # v_map3: 0:[2,-10], 1:[3,-11], 2:[6,-14], 3:[7,-15]  # B12 - B22
    # c_map3: 0:[1], 1:[3], 2:[5], 3:[7]  # C01
    
    # Step 4:
    # u_map4: 0:[10], 1:[11], 2:[14], 3:[15]
    # v_map4: 0:[-0,8], 1:[-1,9], 2:[-4,12], 3:[-5,13]  # B21 - B11
    # c_map4: 0:[8], 1:[9], 2:[12], 3:[13]  # C10
    
    # Step 5:
    # u_map5: 0:[0,2], 1:[1,3], 2:[4,6], 3:[5,7]
    # v_map5: 0:[10], 1:[11], 2:[14], 3:[15]
    # c_map5: 0:[0], 1:[1], 2:[4], 3:[5]  # C00
    
    # Step 6:
    # u_map6: 0:[-0,8], 1:[-1,9], 2:[-4,12], 3:[-5,13]  # A21 - A11
    # v_map6: 0:[0,2], 1:[1,3], 2:[4,6], 3:[5,7]
    # c_map6: 0:[10], 1:[11], 2:[14], 3:[15]  # C11
    
    # Step 7:
    # u_map7: 0:[2,-10], 1:[3,-11], 2:[6,-14], 3:[7,-15]  # A12 - A22
    # v_map7: 0:[8,10], 1:[9,11], 2:[12,14], 3:[13,15]
    # c_map7: 0:[0], 1:[1], 2:[4], 3:[5]  # C00
    
    # Now we can generate the 49 terms.
    # For each step s (1..7), we have 7 Strassen terms.
    # For each term r (1..7), we add the rank-1 tensor.
    
    # Let's implement this.
    
    steps = [
        ([ [0,10], [1,11], [4,14], [5,15] ], [ [0,10], [1,11], [4,14], [5,15] ], [ [0], [1], [4], [5] ]),
        ([ [8,10], [9,11], [12,14], [13,15] ], [ [0], [1], [4], [5] ], [ [8], [9], [12], [13] ]),
        ([ [0], [1], [4], [5] ], [ [2,-10], [3,-11], [6,-14], [7,-15] ], [ [1], [3], [5], [7] ]),
        ([ [10], [11], [14], [15] ], [ [-0,8], [-1,9], [-4,12], [-5,13] ], [ [8], [9], [12], [13] ]),
        ([ [0,2], [1,3], [4,6], [5,7] ], [ [10], [11], [14], [15] ], [ [0], [1], [4], [5] ]),
        ([ [-0,8], [-1,9], [-4,12], [-5,13] ], [ [0,2], [1,3], [4,6], [5,7] ], [ [10], [11], [14], [15] ]),
        ([ [2,-10], [3,-11], [6,-14], [7,-15] ], [ [8,10], [9,11], [12,14], [13,15] ], [ [0], [1], [4], [5] ]),
    ]
    
    # Now generate the 49 terms.
    # Each term is (u_vec, v_vec, w_vec) for the 2x2 case.
    # We need to map them to 4x4.
    
    # u_4x4 is a vector of length 16.
    # u_4x4[idx] += coeff
    
    # For each step s:
    # For each term r in strassen_2x2_terms:
    #   u_vec, v_vec, w_vec = strassen_2x2_terms[r]
    #   u_4x4 = [0]*16
    #   for i in range(4):
    #       for idx in steps[s][0][i]:
    #           u_4x4[idx] += u_vec[i]
    #   Similarly for v and w.
    #   Add this rank-1 term to the global tensor.
    
    # But wait, the rank-1 term is u \otimes v \otimes w.
    # So U_{i, col} = u_4x4[i]
    # V_{j, col} = v_4x4[j]
    # W_{k, col} = w_4x4[k]
    
    # We need to do this for both B1 and B2.
    # For B1, the V indices are 0..15.
    # For B2, the V indices are 16..31.
    # The W indices for B1 are 0..15.
    # The W indices for B2 are 16..31.
    
    # So we can just call the generation twice, with an offset for V and W.
    
    def generate_4x4(offset_v, offset_w):
        col = 0
        for s in range(7):
            for r in range(7):
                u_vec, v_vec, w_vec = strassen_2x2_terms[r]
                u_4x4 = [0.0] * 16
                v_4x4 = [0.0] * 16
                w_4x4 = [0.0] * 16
                
                for i in range(4):
                    for idx in steps[s][0][i]:
                        u_4x4[idx] += u_vec[i]
                for j in range(4):
                    for idx in steps[s][1][j]:
                        v_4x4[idx] += v_vec[j]
                for k in range(4):
                    for idx in steps[s][2][k]:
                        w_4x4[idx] += w_vec[k]
                
                # Add to global tensor
                for i in range(16):
                    U[i][col] = u_4x4[i]
                for j in range(16):
                    V[j + offset_v][col] = v_4x4[j]
                for k in range(16):
                    W[k + offset_w][col] = w_4x4[k]
                
                col += 1
        return col
    
    col1 = generate_4x4(0, 0)
    col2 = generate_4x4(16, 16)
    
    return {'U': U, 'V': V, 'W': W}

