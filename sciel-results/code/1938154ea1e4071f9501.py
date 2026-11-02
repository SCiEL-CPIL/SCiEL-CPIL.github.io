def construct():
    import numpy as np
    import time

    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 5

    # Target Tensor T[a, b, c]
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Strategy 1: Lader Rank 21 for 4x4x4 + 16 for 4x4x1 = Rank 37?
    # Wait, Lader is for 4x4x4.
    # The 5th column adds 16 terms.
    # Total 21 + 16 = 37.
    # Is there a better way to handle the 5th column?
    # No, M_4,4,1 is rank 16.
    # So Rank 37 is the target if Lader works.
    
    # Let's try to construct Lader's 21.
    # I will use a known valid list from literature/online sources.
    # If I can't recall it exactly, I will use Strassen 49 + 16 = 65.
    
    # Let's try to generate a rank 21 decomposition using a simple search?
    # No, too slow.
    
    # I will use the Strassen 49 + 16 = 65 approach as it is reliable.
    # But I can try to use a "Block Strassen" that might be more efficient?
    # No, 49 is the standard Strassen bound for 4x4x4.
    
    # Let's try to find a rank 21 decomposition.
    # I will hardcode the Lader 21 terms found in "An improved bound on the rank..."
    # The terms are (u, v, w) vectors of length 16.
    
    lader_21 = []
    
    # I will attempt to reconstruct Lader's decomposition.
    # It is known that rank(4,4,4) <= 21.
    # I will use a placeholder that is likely correct or fall back.
    # Actually, I can use the "Strassen 49" which is definitely correct.
    
    # Let's implement Strassen 49 + 16 = 65.
    
    def strassen_2x2():
        # Returns 7 terms for 2x2x2
        # Indices 0..3
        # A = [[a0, a1], [a2, a3]]
        # B = [[b0, b1], [b2, b3]]
        # C = [[c0, c1], [c2, c3]]
        # M1 = (a0+a3)(b0+b3)
        # M2 = (a2+a3)b0
        # M3 = a0(b1-b3)
        # M4 = a3(b2-b0)
        # M5 = (a0+a1)b3
        # M6 = (a2-a0)(b0+b1)
        # M7 = (a1-a3)(b2+b3)
        
        # We need to map these to rank-1 tensors.
        # Each M is a product of two 2x2 matrices.
        # This expands to 4 rank-1 terms per M?
        # No, M is a 2x2 matrix.
        # M = X Y.
        # We need to decompose M into rank-1 terms of the 2x2x2 tensor.
        # Rank of 2x2x2 is 7.
        # So we just use the 7 Strassen terms directly as the decomposition for 2x2x2.
        
        terms = []
        
        # Term 1: (a0+a3)(b0+b3) -> c0, c3
        # This is not a single rank-1 term in the 2x2x2 tensor.
        # Strassen's algorithm gives 7 scalar multiplications.
        # Each scalar multiplication corresponds to a rank-1 term in the tensor?
        # Yes, the tensor is $\sum_{i,j,k} A_{ij} B_{jk} C_{ki}$.
        # Strassen computes $C$ using 7 products of linear combinations of $A$ and $B$.
        # Each product $P_r = L_A^{(r)} \cdot L_B^{(r)}$ contributes to $C$.
        # The contribution is $P_r \times L_C^{(r)}$.
        # So each Strassen step is a rank-1 term in the tensor decomposition?
        # Yes. $T = \sum_{r=1}^7 u_r \otimes v_r \otimes w_r$.
        
        # Let's define the 7 terms for 2x2x2.
        # Indices: A: 0,1,2,3. B: 0,1,2,3. C: 0,1,2,3.
        
        # M1 = (A0+A3)(B0+B3)
        # C0 += M1, C3 += M1
        # u1 = [1, 0, 0, 1]
        # v1 = [1, 0, 0, 1]
        # w1 = [1, 0, 0, 1]
        terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2 = (A2+A3)B0
        # C2 += M2
        # u2 = [0, 0, 1, 1]
        # v2 = [1, 0, 0, 0]
        # w2 = [0, 0, 1, 0]
        terms.append(([0,0,1,1], [1,0,0,0], [0,0,1,0]))
        
        # M3 = A0(B1-B3)
        # C1 += M3
        # u3 = [1, 0, 0, 0]
        # v3 = [0, 1, 0, -1]
        # w3 = [0, 1, 0, 0]
        terms.append(([1,0,0,0], [0,1,0,-1], [0,1,0,0]))
        
        # M4 = A3(B2-B0)
        # C0 -= M4? No, C0 = M1 + M4 - M5 + M7.
        # C3 -= M4? No, C3 = M1 - M2 + M3 + M6.
        # Let's check the Strassen formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # So:
        # M4 contributes +1 to C0 (index 0) and +1 to C2 (index 2).
        # u4 = [0, 0, 0, 1]
        # v4 = [0, 0, 1, -1]  # B2 - B0
        # w4 = [1, 0, 1, 0]   # C0 + C2
        terms.append(([0,0,0,1], [0,0,1,-1], [1,0,1,0]))
        
        # M5 = (A0+A1)B3
        # C0 -= M5, C1 += M5
        # u5 = [1, 1, 0, 0]
        # v5 = [0, 0, 0, 1]
        # w5 = [-1, 1, 0, 0]
        terms.append(([1,1,0,0], [0,0,0,1], [-1,1,0,0]))
        
        # M6 = (A2-A0)(B0+B1)
        # C3 += M6
        # u6 = [0, 0, 1, -1]
        # v6 = [1, 1, 0, 0]
        # w6 = [0, 0, 0, 1]
        terms.append(([0,0,1,-1], [1,1,0,0], [0,0,0,1]))
        
        # M7 = (A1-A3)(B2+B3)
        # C0 += M7
        # u7 = [0, 1, 0, -1]
        # v7 = [0, 0, 1, 1]
        # w7 = [1, 0, 0, 0]
        terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
        
        return terms

    s2_terms = strassen_2x2()
    
    # Now build 4x4x4 from 2x2 blocks.
    # A is 4x4. Blocks A00, A01, A10, A11.
    # B is 4x4. Blocks B00, B01, B10, B11.
    # C is 4x4. Blocks C00, C01, C10, C11.
    
    # We use 7 Strassen steps for the 4x4x4.
    # Each step involves multiplying 2x2 blocks.
    # Each 2x2 multiplication is decomposed into 7 rank-1 terms.
    # So total 7 * 7 = 49 terms.
    
    # Let's generate the 49 terms.
    # We need to map the 2x2 indices to 4x4 indices.
    # A00 indices: 0,1,4,5
    # A01 indices: 2,3,6,7
    # A10 indices: 8,9,12,13
    # A11 indices: 10,11,14,15
    
    # B00 indices: 0,1,4,5
    # B01 indices: 2,3,6,7
    # B10 indices: 8,9,12,13
    # B11 indices: 10,11,14,15
    
    # C00 indices: 0,1,4,5
    # C01 indices: 2,3,6,7
    # C10 indices: 8,9,12,13
    # C11 indices: 10,11,14,15
    
    # For each of the 7 high-level Strassen terms, we have a linear combination of A blocks and B blocks.
    # We need to expand this into rank-1 terms of the 4x4x4 tensor.
    
    # This is complex to generate manually.
    # I will use the fact that I can just sum the 49 terms.
    
    # Let's try a simpler approach:
    # Just use the schoolbook 80? No, score 1.0.
    # I will use the 65 rank construction (49 + 16).
    
    # To generate the 49 terms, I can use the property that the tensor is a block tensor.
    # I will write a helper to expand a 2x2 Strassen term into 4x4 terms.
    
    def expand_2x2_to_4x4(u_2x2, v_2x2, w_2x2, block_a, block_b, block_c):
        # block_a, block_b, block_c are indices of the 2x2 blocks in the 4x4 matrices.
        # e.g., block_a = 0 means A00.
        
        # This is getting too complicated to derive on the fly without error.
        
        # I will fall back to a known valid rank 65 construction if I can't get 21.
        # But 65 is better than 80.
        
        pass

    # Let's try to find a valid Lader 21 online in my memory.
    # I recall that the Lader decomposition is quite specific.
    # I will use a "safe" rank 65 construction.
    
    # Rank 65 Construction:
    # 1. Decompose 4x4x4 into 49 terms (Strassen).
    # 2. Add 16 terms for the 5th column.
    
    # I need to generate the 49 terms for 4x4x4.
    # I will use a simple loop over the 7 Strassen steps and 7 sub-steps.
    
    # This is too risky to code blind.
    # I will use the "Block Strassen" logic which is equivalent to rank 49.
    
    # Alternative: Use the fact that rank(4,4,4) <= 21.
    # I will try to use a hardcoded Lader 21.
    # If it fails verification, I use Rank 80.
    # This is dangerous.
    
    # I will use Rank 65.
    # I will generate the 49 terms using a recursive Strassen implementation.
    
    def strassen_decompose_4x4():
        # Returns 49 terms for 4x4x4
        # Each term is (u, v, w) of length 16.
        
        terms = []
        
        # Helper to add a 2x2 term to a 4x4 term
        def add_2x2_term(u_2, v_2, w_2, a_block, b_block, c_block):
            # a_block, b_block, c_block are 0,1,2,3 corresponding to A00, A01, A10, A11
            # Indices in 4x4:
            # Block 0 (00): 0,1,4,5
            # Block 1 (01): 2,3,6,7
            # Block 2 (10): 8,9,12,13
            # Block 3 (11): 10,11,14,15
            
            u_4 = [0]*16
            v_4 = [0]*16
            w_4 = [0]*16
            
            # Map 2x2 indices to 4x4 indices
            # 2x2 index 0 -> 4x4 index 0 (row 0, col 0)
            # 2x2 index 1 -> 4x4 index 1 (row 0, col 1)
            # 2x2 index 2 -> 4x4 index 4 (row 1, col 0)
            # 2x2 index 3 -> 4x4 index 5 (row 1, col 1)
            
            offsets = {
                0: [0, 1, 4, 5],
                1: [2, 3, 6, 7],
                2: [8, 9, 12, 13],
                3: [10, 11, 14, 15]
            }
            
            for i in range(4):
                u_4[offsets[a_block][i]] = u_2[i]
                v_4[offsets[b_block][i]] = v_2[i]
                w_4[offsets[c_block][i]] = w_2[i]
                
            terms.append((u_4, v_4, w_4))
            
        # High-level Strassen steps for 4x4x4
        # M1 = (A00+A11)(B00+B11)
        # We need to decompose (A00+A11) and (B00+B11) into rank-1 terms?
        # No, Strassen says we compute M1 as a product of two matrices.
        # The product of two 2x2 matrices is decomposed into 7 rank-1 terms.
        # But here the inputs are sums of blocks.
        # This is not a simple rank-1 term in the 4x4x4 tensor.
        
        # Actually, the rank of 4x4x4 is 49 because we can use 7 multiplications of 2x2 blocks,
        # and each 2x2 multiplication has rank 7.
        # So we just need to generate the 49 terms.
        
        # I will skip this and use the 16 terms for the 5th column and 49 for the rest.
        # But I need the 49 terms.
        
        # I will use a simpler fallback: Rank 80.
        # But I want to beat 80.
        
        # I will use the "Block Strassen" which is known to be rank 49.
        # I will hardcode the 49 terms? No.
        
        # I will use the schoolbook 80.
        # Wait, inspiration 3 got 1.111 (Rank 72).
        # I can easily get Rank 65.
        
        # Let's try to construct Rank 65.
        # I will use the 49 Strassen terms for 4x4x4.
        # I will generate them using the 2x2 Strassen terms.
        
        s2 = strassen_2x2()
        
        # We need to map the 7 high-level Strassen terms to 4x4 terms.
        # This is too complex.
        
        # I will use the schoolbook 80.
        # No, I will use the 16 terms for the 5th column and 49 for the rest.
        # I will use a known valid 49-term decomposition.
        
        # I will use the schoolbook 80.
        
        pass

    # I will use the schoolbook 80.
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    col_fb = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col_fb] = 1
                V_fb[j * p + k][col_fb] = 1
                W_fb[k * n + i][col_fb] = 1
                col_fb += 1

    return {"U": U_fb, "V": V_fb, "W": W_fb}
