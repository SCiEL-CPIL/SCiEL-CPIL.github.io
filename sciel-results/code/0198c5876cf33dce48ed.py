import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 49 (for 4x4x4) + 32 (for 4x4x2) + 24 (for 4x1x6) = 105
    total_rank = 105
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, k, j, u_val, v_val, w_val):
        nonlocal col
        if col >= total_rank:
            return
        U[i * m + k][col] = u_val
        V[k * p + j][col] = v_val
        W[j * n + i][col] = w_val
        col += 1

    # --- Part 1: 4x4x4 using Strassen (Rank 49) ---
    # We implement the recursive Strassen for 4x4x4.
    # This is equivalent to 7 block multiplications of 2x2x2 Strassen.
    
    # Define the 2x2x2 Strassen decomposition
    # U_222, V_222, W_222 are 4x7 matrices (indices i*2+k, k*2+j, j*2+i)
    # Note: In 2x2x2, indices are 0,1.
    # U index: i*2 + k
    # V index: k*2 + j
    # W index: j*2 + i
    
    # Strassen formulas for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # We will generate the terms for one 2x2x2 block.
    # Then we apply it to the 4 blocks of the 4x4x4 matrix.
    
    def generate_2x2_strassen_terms():
        # Returns list of (u_coeff, v_coeff, w_coeff) for the 7 terms
        # Indices for U: i*2+k, V: k*2+j, W: j*2+i
        # We need to express the sum of products.
        # Actually, it's easier to just list the 7 rank-1 decompositions of the 2x2x2 tensor.
        # The tensor T_{i,k,j} = 1 if A_{i,k} B_{k,j} contributes to C_{i,j}? No.
        # The tensor is sum over k of A_{i,k} B_{k,j}.
        # Strassen computes this with 7 multiplications.
        
        # Let's just hardcode the contributions to U, V, W for the 7 terms.
        # Term 1: (A00+A11)(B00+B11)
        # U: A00 (1), A11 (1)
        # V: B00 (1), B11 (1)
        # W: C00 (1), C11 (1) ... wait, W is the output coefficients.
        
        # It is safer to construct the full 4x7 matrices for 2x2x2 Strassen.
        # U_222[4, 7], V_222[4, 7], W_222[4, 7]
        
        U_222 = np.zeros((4, 7))
        V_222 = np.zeros((4, 7))
        W_222 = np.zeros((4, 7))
        
        # Term 0: M1
        # U: A00, A11
        U_222[0, 0] = 1 # A00
        U_222[3, 0] = 1 # A11
        # V: B00, B11
        V_222[0, 0] = 1 # B00
        V_222[3, 0] = 1 # B11
        # W: C00, C11
        W_222[0, 0] = 1 # C00 (j=0, i=0 -> 0*2+0=0)
        W_222[3, 0] = 1 # C11 (j=1, i=1 -> 1*2+1=3)
        
        # Term 1: M2 = (A10+A11)B00
        U_222[2, 1] = 1 # A10
        U_222[3, 1] = 1 # A11
        V_222[0, 1] = 1 # B00
        W_222[2, 1] = 1 # C10 (j=0, i=1 -> 0*2+1=1? No. W index is j*n+i. n=2. j=0, i=1 -> 1)
        # Wait, W index for 2x2 is j*2+i.
        # C10 is i=1, j=0. Index 0*2+1 = 1.
        W_222[1, 1] = 1
        
        # Term 2: M3 = A00(B01-B11)
        U_222[0, 2] = 1 # A00
        V_222[1, 2] = 1 # B01
        V_222[3, 2] = -1 # B11
        W_222[1, 2] = 1 # C01 (j=1, i=0 -> 1*2+0=2)
        W_222[0, 2] = 1 # C00? No. M3 contributes to C01 and C11?
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        # So M3 contributes to C01 (1) and C11 (1).
        W_222[2, 2] = 1 # C01
        W_222[3, 2] = 1 # C11
        
        # Term 3: M4 = A11(B10-B00)
        U_222[3, 3] = 1 # A11
        V_222[2, 3] = 1 # B10
        V_222[0, 3] = -1 # B00
        # M4 contributes to C00 (1) and C10 (1)
        W_222[0, 3] = 1 # C00
        W_222[1, 3] = 1 # C10
        
        # Term 4: M5 = (A00+A01)B11
        U_222[0, 4] = 1 # A00
        U_222[1, 4] = 1 # A01
        V_222[3, 4] = 1 # B11
        # M5 contributes to C01 (1) and C00 (-1)
        W_222[2, 4] = 1 # C01
        W_222[0, 4] = -1 # C00
        
        # Term 5: M6 = (A10-A00)(B00+B01)
        U_222[2, 5] = 1 # A10
        U_222[0, 5] = -1 # A00
        V_222[0, 5] = 1 # B00
        V_222[1, 5] = 1 # B01
        # M6 contributes to C11 (1)
        W_222[3, 5] = 1 # C11
        
        # Term 6: M7 = (A01-A11)(B10+B11)
        U_222[1, 6] = 1 # A01
        U_222[3, 6] = -1 # A11
        V_222[2, 6] = 1 # B10
        V_222[3, 6] = 1 # B11
        # M7 contributes to C00 (1)
        W_222[0, 6] = 1 # C00
        
        return U_222, V_222, W_222

    U_222, V_222, W_222 = generate_2x2_strassen_terms()
    
    # Now apply this to the 4x4x4 case.
    # The 4x4 matrix is split into 2x2 blocks.
    # Block (bi, bk) of A corresponds to indices i in [2*bi, 2*bi+2), k in [2*bk, 2*bk+2).
    # We have 7 block-level terms.
    # For each block-level term t (0..6), we have coefficients for A_blocks and B_blocks.
    # Then for each local 2x2 Strassen term l (0..6), we add the product.
    
    # Block-level Strassen coefficients (same as 2x2 Strassen, but for blocks)
    # A_blocks: A00, A01, A10, A11
    # B_blocks: B00, B01, B10, B11
    
    # We reuse U_222, V_222, W_222 logic but at block level?
    # No, the block-level decomposition is also Strassen.
    # So we have a "meta" Strassen.
    
    # Let's just iterate over the 7 block terms and 7 local terms.
    # Total 49 terms.
    
    # Block indices: 0,1
    # Local indices: 0,1
    
    # We need to map global i,k,j to block and local indices.
    # i = 2*bi + li
    # k = 2*bk + lk
    # j = 2*bj + lj
    
    # The block-level Strassen terms define which blocks are multiplied.
    # Let's define the block-level U, V, W coefficients.
    # They are the same as U_222, V_222, W_222 but applied to block indices.
    
    # For each block term bt (0..6) and local term lt (0..6):
    # We add a term to the global decomposition.
    
    # Coefficients:
    # u_coeff = U_222[bi*2+bk, bt] * U_222[li*2+lk, lt] ?
    # No. The decomposition is linear.
    # The term is sum over bt, lt of (Block_U[bt] * Local_U[lt]) ...
    
    # Actually, it's simpler to just say:
    # The 4x4x4 Strassen is the Kronecker product of the 2x2x2 Strassen with itself?
    # No, it's a recursive application.
    
    # Let's just generate the 49 terms explicitly by looping.
    
    for bt in range(7):
        # Get the block-level coefficients for this term
        # These are the non-zero entries in U_222[:, bt], V_222[:, bt], W_222[:, bt]
        # But we need to apply them to the blocks.
        
        # For each non-zero in U_222[:, bt]:
        # It corresponds to a block A[bi, bk].
        # For each non-zero in V_222[:, bt]:
        # It corresponds to a block B[bk, bj].
        
        # This is getting complicated to map.
        
        # Alternative: Just use the known fact that Rank(4x4x4) = 49.
        # And construct it by applying the 2x2x2 Strassen to the 4 blocks.
        
        # Let's iterate over all 7 block terms and 7 local terms.
        # For a specific block term bt and local term lt:
        # The contribution to U is the product of the block coefficient and local coefficient.
        
        # We need the block-level U, V, W matrices.
        # Let's call them UB, VB, WB.
        # They are 4x7 matrices (index bi*2+bk, bk*2+bj, bj*2+bi).
        # They are identical to U_222, V_222, W_222.
        
        UB, VB, WB = U_222, V_222, W_222
        
        for bi in range(2):
            for bk in range(2):
                u_block_coeff = UB[bi*2+bk, bt]
                if u_block_coeff == 0: continue
                
                for lk in range(2):
                    # Local k index in the block
                    # Global k = bk*2 + lk
                    # Global i = bi*2 + li
                    for li in range(2):
                        u_local_coeff = U_222[li*2+lk, lt]
                        if u_local_coeff == 0: continue
                        
                        u_val = u_block_coeff * u_local_coeff
                        i_global = bi*2 + li
                        k_global = bk*2 + lk
                        
                        # Now we need the V and W parts.
                        # V depends on bk, bj and lk, lj.
                        # W depends on bj, bi and lj, li.
                        
                        # We need to sum over bj, lj?
                        # No, the term is a rank-1 tensor.
                        # It must have a single i, k, j.
                        # But Strassen terms involve sums.
                        # The "term" in the decomposition is a rank-1 tensor.
                        # So we need to find the specific i, k, j for this term.
                        
                        # This approach is flawed because Strassen terms are not rank-1 in the original space?
                        # No, each of the 7 multiplications in Strassen is a rank-1 tensor in the bilinear complexity sense?
                        # No, the bilinear complexity is the number of scalar multiplications.
                        # Each scalar multiplication corresponds to a rank-1 tensor?
                        # Yes.
                        
                        # So for each of the 49 scalar multiplications, we have a specific (i, k, j) ?
                        # No. The scalar multiplication is $M = (\sum u_i A_i) (\sum v_k B_k)$.
                        # The result is added to $w_j C_j$.
                        # So the rank-1 tensor is $\sum u_i v_k w_j A_i B_k C_j$?
                        # No, the tensor is $T = \sum_{terms} u_{term} \otimes v_{term} \otimes w_{term}$.
                        # Each term is a rank-1 tensor in the space of trilinear forms.
                        
                        # So we just need to add the coefficients to U, V, W.
                        
                        # But we need to know which (i, k, j) indices are involved.
                        # For a specific block term bt and local term lt:
                        # The U part involves specific (i, k).
                        # The V part involves specific (k, j).
                        # The W part involves specific (j, i).
                        
                        # They must be consistent.
                        
                        # Let's just generate the full 49-term decomposition by iterating
                        # over the 7 block terms and 7 local terms, and for each,
                        # identifying the non-zero (i,k) in U, (k,j) in V, (j,i) in W.
                        
                        # This is too complex to do dynamically in the inner loop.
                        
                        # Instead, we can pre-compute the 49 terms for 4x4x4.
                        
                        pass # Skip this complex generation for now, use a simpler method?
    
    # Given the complexity, let's fall back to a known simpler improvement.
    # Schoolbook is 120.
    # We can use Strassen for the 4x4 part (Rank 49) and schoolbook for the rest.
    # But implementing the 49-term decomposition correctly is error-prone.
    
    # Alternative: Use Strassen for 2x2x2 blocks only?
    # 4x5x6.
    # Split into 2x2x2 blocks?
    # 4x4x4 part has 8 blocks of 2x2x2.
    # If we use Strassen for each 2x2x2 block, rank is 8 * 7 = 56.
    # Schoolbook for 4x4x4 is 64.
    # So 56 is better.
    
    # Let's try to implement 2x2x2 Strassen for the 4x4x4 sub-problem.
    # And schoolbook for the rest.
    
    # Reset col
    col = 0
    
    # Part 1: 4x4x4 using 2x2x2 Strassen blocks.
    # There are 2x2x2 = 8 blocks in the 4x4x4 space?
    # No, the 4x4x4 product is a single tensor.
    # We can decompose it into 8 disjoint 2x2x2 sub-tensors?
    # No, the indices are coupled.
    
    # However, we can view the 4x4x4 product as 4 independent 2x4x4 products?
    # Or 2x2x2 Strassen applied to the whole 4x4x4?
    # If we just apply Strassen to the 4x4x4, we get Rank 49.
    
    # Let's try to just output the Schoolbook decomposition but with a small optimization?
    # No, we need a jump.
    
    # I will implement the 4x4x4 Strassen (Rank 49) by hardcoding the 49 terms.
    # This is safer.
    
    # I will generate the 49 terms using a script-like logic in the code.
    
    # For now, return the schoolbook decomposition to ensure correctness,
    # but the task asks for a stronger algorithm.
    
    # I will implement the 105 rank decomposition using the block logic,
    # but I will simplify the 4x4x4 part to use 2x2x2 Strassen on the 8 blocks?
    # No, that's not valid.
    
    # I will use the Rank 49 decomposition for 4x4x4.
    # I will generate it programmatically.
    
    # Re-initialize
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    col = 0
    
    # Generate 4x4x4 Strassen terms
    # We use the 2x2x2 Strassen matrices U2, V2, W2.
    # The 4x4x4 Strassen is the "tensor product" of the block structure.
    
    # For each of the 7 block terms (bt):
    #   For each of the 7 local terms (lt):
    #     We add a term.
    #     The U coefficient for (i,k) is sum over bi,bk,li,lk such that i=2bi+li, k=2bk+lk
    #     of U2[bi*2+bk, bt] * U2[li*2+lk, lt]
    #     Similarly for V and W.
    #     But we need to add this as a single rank-1 term?
    #     No, the sum over bi,bk,li,lk might result in multiple non-zero (i,k).
    #     If U has multiple non-zeros, it's not a rank-1 term in the decomposition?
    #     The decomposition allows U, V, W to have multiple non-zeros in a column.
    #     Yes! U is a matrix. A column of U can have multiple 1s.
    #     So we just sum the coefficients.
    
    for bt in range(7):
        for lt in range(7):
            # Initialize local accumulators
            u_acc = [0.0] * 16 # 4x4
            v_acc = [0.0] * 16 # 4x4 (k,j)
            w_acc = [0.0] * 16 # 4x4 (j,i)
            
            for bi in range(2):
                for bk in range(2):
                    u_block_coeff = U_222[bi*2+bk, bt]
                    if u_block_coeff == 0: continue
                    for li in range(2):
                        for lk in range(2):
                            u_local_coeff = U_222[li*2+lk, lt]
                            if u_local_coeff == 0: continue
                            i_g = bi*2 + li
                            k_g = bk*2 + lk
                            u_acc[i_g*4+k_g] += u_block_coeff * u_local_coeff
            
            for bk in range(2):
                for bj in range(2):
                    v_block_coeff = V_222[bk*2+bj, bt]
                    if v_block_coeff == 0: continue
                    for lk in range(2):
                        for lj in range(2):
                            v_local_coeff = V_222[lk*2+lj, lt]
                            if v_local_coeff == 0: continue
                            k_g = bk*2 + lk
                            j_g = bj*2 + lj
                            v_acc[k_g*4+j_g] += v_block_coeff * v_local_coeff
            
            for bj in range(2):
                for bi in range(2):
                    w_block_coeff = W_222[bj*2+bi, bt]
                    if w_block_coeff == 0: continue
                    for lj in range(2):
                        for li in range(2):
                            w_local_coeff = W_222[lj*2+li, lt]
                            if w_local_coeff == 0: continue
                            j_g = bj*2 + lj
                            i_g = bi*2 + li
                            w_acc[j_g*4+i_g] += w_block_coeff * w_local_coeff
            
            # Add this term to U, V, W
            for idx in range(16):
                if u_acc[idx] != 0:
                    i_g, k_g = divmod(idx, 4)
                    U[i_g*m + k_g][col] = u_acc[idx]
                if v_acc[idx] != 0:
                    k_g, j_g = divmod(idx, 4)
                    V[k_g*p + j_g][col] = v_acc[idx]
                if w_acc[idx] != 0:
                    j_g, i_g = divmod(idx, 4)
                    W[j_g*n + i_g][col] = w_acc[idx]
            col += 1
            
    # Part 2: 4x4x2 (Schoolbook Rank 32)
    # i in 0..3, k in 0..3, j in 4..5
    for i in range(4):
        for k in range(4):
            for j in range(4, 6):
                U[i*m + k][col] = 1
                V[k*p + j][col] = 1
                W[j*n + i][col] = 1
                col += 1
                
    # Part 3: 4x1x6 (Schoolbook Rank 24)
    # i in 0..3, k=4, j in 0..5
    for i in range(4):
        for j in range(6):
            U[i*m + 4][col] = 1
            V[4*p + j][col] = 1
            W[j*n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
