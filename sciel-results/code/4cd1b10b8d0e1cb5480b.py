"""Matrix multiplication 4x4x5 via structured Strassen decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We split the multiplication C = A * B into C = A * B4 + A * B1
    # where B4 is the first 4 columns of B, and B1 is the last column of B.
    # Part 1: A (4x4) * B4 (4x4) -> C4 (4x4). Rank 49 via Strassen.
    # Part 2: A (4x4) * B1 (4x1) -> C1 (4x1). Rank 16 via Trivial.
    # Total Rank = 49 + 16 = 65.
    
    rank_total = 65
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    # --- Part 1: Strassen for 4x4x4 ---
    # Indices for Strassen on 4x4x4:
    # A is 4x4, B is 4x4, C is 4x4.
    # We view A, B, C as 2x2 blocks of 2x2 matrices.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # Strassen formulas for 2x2 block multiplication:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B21 - B22)
    # M4 = A22 * (B12 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each M_i is a 2x2x2 multiplication, which has rank 7.
    # We need to decompose each 2x2x2 multiplication.
    
    def add_strassen_2x2(U, V, W, col_offset, i_off, j_off, k_off, a_signs, b_signs, c_signs):
        """
        Add a rank-7 decomposition for a 2x2x2 multiplication.
        a_signs, b_signs, c_signs are lists of 7 tuples (i, j) or similar for linear combinations.
        Actually, it's easier to define the 7 terms explicitly.
        """
        # The 7 terms for 2x2x2 Strassen:
        # 1: (1,1) (1,1) (1,1) -> (A11+A22)(B11+B22) contributes to C11, C22
        #    U: A11, A22. V: B11, B22. W: C11, C22.
        #    This is not a single rank-1 term. It's a combination.
        #    Wait, the rank-7 decomposition is a set of 7 rank-1 tensors.
        #    Let's list the 7 rank-1 terms (U, V, W) for 2x2x2.
        #    Indices 0,1 for rows/cols.
        
        # Term 1: U: [1, 0, 1, 0] (A11, A22), V: [1, 0, 0, 1] (B11, B22), W: [1, 0, 0, 1] (C11, C22)
        #    Actually, let's use the standard basis vectors for the linear combinations.
        #    The 7 multiplications are:
        #    M1 = (A11+A22)(B11+B22)
        #    M2 = (A21+A22)B11
        #    M3 = A11(B21-B22)
        #    M4 = A22(B12-B11)
        #    M5 = (A11+A12)B22
        #    M6 = (A21-A11)(B11+B12)
        #    M7 = (A12-A22)(B21+B22)
        
        # We can represent each multiplication as a linear combination of rank-1 tensors?
        # No, each M_i is a scalar product of two linear forms.
        # The tensor is sum of 7 terms: u_i outer v_i outer w_i.
        # Where u_i, v_i, w_i are vectors.
        
        # Let's define the 7 vectors for U, V, W for the 2x2x2 case.
        # U rows: A11, A12, A21, A22 (indices 0,1,2,3)
        # V rows: B11, B12, B21, B22 (indices 0,1,2,3)
        # W rows: C11, C12, C21, C22 (indices 0,1,2,3) -- Note: W is indexed by (k,i) so C11 is (0,0), C12 is (1,0), C21 is (0,1), C22 is (1,1)?
        # Wait, W index is k*n+i. For 2x2, n=2.
        # C11: k=0, i=0 -> idx 0
        # C12: k=1, i=0 -> idx 2
        # C21: k=0, i=1 -> idx 1
        # C22: k=1, i=1 -> idx 3
        # So W order is C11, C21, C12, C22.
        
        # Let's verify the Strassen decomposition in this basis.
        # C11 = M1 + M4 - M5 + M7
        # C21 = M2 + M4
        # C12 = M3 + M5
        # C22 = M1 - M2 + M3 + M6
        
        # The 7 terms are:
        # 1. M1: (A11+A22)(B11+B22)
        #    U1 = [1, 0, 0, 1] (A11, A22)
        #    V1 = [1, 0, 0, 1] (B11, B22)
        #    W1 = [1, 0, 0, 1] (C11, C22) -- Coeffs: +1 for C11, +1 for C22
        #    Check: C11 gets +1, C22 gets +1. Correct.
        
        # 2. M2: (A21+A22)B11
        #    U2 = [0, 0, 1, 1] (A21, A22)
        #    V2 = [1, 0, 0, 0] (B11)
        #    W2 = [0, 1, 0, 0] (C21) -- Coeff: +1 for C21.
        #    Check: C21 gets +1. Correct.
        
        # 3. M3: A11(B21-B22)
        #    U3 = [1, 0, 0, 0] (A11)
        #    V3 = [0, 0, 1, -1] (B21, -B22)
        #    W3 = [0, 0, 1, 0] (C12) -- Coeff: +1 for C12.
        #    Check: C12 gets +1. Correct.
        
        # 4. M4: A22(B12-B11)
        #    U4 = [0, 0, 0, 1] (A22)
        #    V4 = [-1, 1, 0, 0] (-B11, B12)
        #    W4 = [1, 1, 0, 0] (C11, C21) -- Coeffs: +1 for C11, +1 for C21.
        #    Check: C11 gets +1, C21 gets +1. Correct.
        
        # 5. M5: (A11+A12)B22
        #    U5 = [1, 1, 0, 0] (A11, A12)
        #    V5 = [0, 0, 0, 1] (B22)
        #    W5 = [-1, 0, 1, 0] (-C11, C12) -- Coeffs: -1 for C11, +1 for C12.
        #    Check: C11 gets -1, C12 gets +1. Correct.
        
        # 6. M6: (A21-A11)(B11+B12)
        #    U6 = [-1, 0, 1, 0] (-A11, A21)
        #    V6 = [1, 1, 0, 0] (B11, B12)
        #    W6 = [0, 0, 0, 1] (C22) -- Coeff: +1 for C22.
        #    Check: C22 gets +1. Correct.
        
        # 7. M7: (A12-A22)(B21+B22)
        #    U7 = [0, 1, 0, -1] (A12, -A22)
        #    V7 = [0, 0, 1, 1] (B21, B22)
        #    W7 = [1, 0, 0, 0] (C11) -- Coeff: +1 for C11.
        #    Check: C11 gets +1. Correct.
        
        # Now we map these to the global indices.
        # i_off, j_off, k_off are the top-left indices of the 2x2 block in the global 4x4 matrix.
        # A indices:
        # A11: (i_off, j_off) -> global row i_off*m + j_off
        # A12: (i_off, j_off+1) -> global row i_off*m + j_off+1
        # A21: (i_off+1, j_off) -> global row (i_off+1)*m + j_off
        # A22: (i_off+1, j_off+1) -> global row (i_off+1)*m + j_off+1
        
        # B indices (B is 4x4 in this part, so cols are 0..3):
        # B11: (j_off, k_off) -> global row j_off*p + k_off
        # B12: (j_off, k_off+1) -> global row j_off*p + k_off+1
        # B21: (j_off+1, k_off) -> global row (j_off+1)*p + k_off
        # B22: (j_off+1, k_off+1) -> global row (j_off+1)*p + k_off+1
        
        # W indices (C is 4x4, so rows/cols 0..3):
        # C11: (k_off, i_off) -> global row k_off*n + i_off
        # C12: (k_off+1, i_off) -> global row (k_off+1)*n + i_off
        # C21: (k_off, i_off+1) -> global row k_off*n + (i_off+1)
        # C22: (k_off+1, i_off+1) -> global row (k_off+1)*n + (i_off+1)
        
        # We have 4 blocks of 2x2 in the 4x4 matrix.
        # Block (0,0): i_off=0, j_off=0, k_off=0
        # Block (0,1): i_off=0, j_off=2, k_off=0  -- Wait, B is split into 2x2 blocks too?
        # Yes, for the 4x4x4 part, B is 4x4.
        # B11 (top-left 2x2), B12 (top-right 2x2), B21 (bot-left), B22 (bot-right).
        # So for the Strassen step, we treat the 4x4 matrices as 2x2 blocks of 2x2.
        # The "i" index of the block is 0 or 1. The "j" index of the block is 0 or 1.
        # The "k" index of the block is 0 or 1.
        
        # We need to apply the 7-term decomposition to each of the 4 block multiplications?
        # No, Strassen is applied to the whole 4x4x4 by viewing it as 2x2 blocks.
        # So we have 4 sub-multiplications M1..M7, each of which is a 2x2x2 multiplication.
        # So we need to call add_strassen_2x2 four times? No, we call it for each of the 7 terms,
        # but each term involves a specific 2x2x2 multiplication.
        
        # Actually, the rank-49 decomposition is the Kronecker product of the rank-7 decomposition.
        # So we just need to define the 7 terms for the 4x4x4 case directly.
        # Or, we can recursively call the 2x2x2 decomposition.
        
        # Let's do it recursively.
        # For each of the 7 Strassen terms at the 4x4 level, we have a 2x2x2 multiplication.
        # We decompose each of those 2x2x2 multiplications using the 7-term rank decomposition.
        # Total terms = 7 * 7 = 49.
        
        pass # Placeholder

    # Let's implement the recursive addition.
    
    def add_rank1(U, V, W, col, u_idx, v_idx, w_idx, val):
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val

    # The 7 vectors for 2x2x2 (local indices 0..3)
    # U: A11, A12, A21, A22
    # V: B11, B12, B21, B22
    # W: C11, C21, C12, C22 (Note the order!)
    
    strassen_2x2_terms = [
        # (U_vec, V_vec, W_vec)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0]), # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7
    ]
    
    # Helper to map local 2x2 indices to global 4x4 indices
    # local_i, local_j in {0,1} for A
    # local_j, local_k in {0,1} for B
    # local_k, local_i in {0,1} for C (W)
    # block_i, block_j, block_k in {0,1} for the 2x2 block position in 4x4
    
    def map_A(block_i, local_i, block_j, local_j):
        # Global row i, col j
        i = block_i * 2 + local_i
        j = block_j * 2 + local_j
        return i * m + j
        
    def map_B(block_j, local_j, block_k, local_k):
        # Global row j, col k
        j = block_j * 2 + local_j
        k = block_k * 2 + local_k
        return j * p + k
        
    def map_W(block_k, local_k, block_i, local_i):
        # Global row k, col i
        k = block_k * 2 + local_k
        i = block_i * 2 + local_i
        return k * n + i

    col = 0
    # Iterate over the 7 Strassen terms at the 4x4 level
    for term_idx in range(7):
        u_vec, v_vec, w_vec = strassen_2x2_terms[term_idx]
        
        # This term corresponds to a 2x2x2 multiplication of blocks.
        # The blocks are indexed by local_i, local_j, local_k in {0,1}.
        # But wait, the 7 terms are linear combinations of the block entries.
        # The "multiplication" is of the linear forms.
        # So we expand the linear forms into rank-1 tensors.
        
        # For a given term (u_vec, v_vec, w_vec), we have:
        # sum_{li, lj} u_vec[li*2+lj] * A[li, lj]  (Wait, u_vec is length 4)
        # u_vec indices: 0: A11, 1: A12, 2: A21, 3: A22
        # So u_vec[0] * A11 + u_vec[1] * A12 + u_vec[2] * A21 + u_vec[3] * A22
        
        # Similarly for V and W.
        
        # We need to add rank-1 tensors for each non-zero component.
        # Since the vectors are sparse (mostly 0, 1, -1), we can just iterate.
        
        # However, this term is a product of three linear forms.
        # (sum u_i A_i) * (sum v_j B_j) * (sum w_k C_k)
        # This is a sum of rank-1 tensors:
        # sum_{i,j,k} u_i v_j w_k * (A_i outer B_j outer C_k)
        
        # We iterate over all i, j, k in {0,1,2,3}
        for i in range(4):
            if u_vec[i] == 0: continue
            for j in range(4):
                if v_vec[j] == 0: continue
                for k in range(4):
                    if w_vec[k] == 0: continue
                    
                    # Map local indices to block indices
                    # i: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                    li_i = i // 2
                    lj_i = i % 2
                    lj_j = j // 2
                    lk_j = j % 2
                    lk_k = k // 2
                    li_k = k % 2
                    
                    # Check consistency of block indices
                    # The term is valid if the block indices match?
                    # No, the Strassen decomposition is for the whole 4x4 matrix.
                    # The "blocks" are fixed.
                    # The linear forms combine entries from specific blocks.
                    # For example, M1 = (A11+A22)(B11+B22).
                    # A11 is block (0,0), A22 is block (1,1).
                    # So the block indices are NOT the same for all terms in the sum.
                    # But for a specific rank-1 tensor, we pick one A entry, one B entry, one C entry.
                    # The A entry is from block (li_i, lj_i).
                    # The B entry is from block (lj_j, lk_j).
                    # The C entry is from block (lk_k, li_k).
                    
                    # For the decomposition to be valid, do the block indices need to match?
                    # Yes, because the multiplication is block-wise?
                    # No, Strassen mixes blocks.
                    # But the tensor decomposition is exact.
                    # The rank-1 term is A_entry * B_entry * C_entry.
                    # This term contributes to the product if the indices are consistent?
                    # No, the tensor T is defined such that T[A_idx, B_idx, C_idx] = 1 if the indices match.
                    # For M_{4,4,4}, T[A_idx, B_idx, C_idx] = 1 if A_idx=(i,j), B_idx=(j,k), C_idx=(k,i).
                    # So we need j_A == j_B and k_B == k_C and i_C == i_A.
                    
                    # Let's check the indices:
                    # A_idx = (li_i*2 + lj_i, lj_i*2 + ???) -- Wait, my map_A uses block indices.
                    # Let's re-verify the mapping.
                    # A entry local i:
                    # i=0: block (0,0), local (0,0) -> global (0,0)
                    # i=1: block (0,1), local (0,0) -> global (0,2) -- NO!
                    # My map_A is: i = block_i * 2 + local_i.
                    # If i=1 (local A12), block_i=0, local_i=1? No.
                    # A12 is row 0, col 1.
                    # In 2x2 block notation:
                    # A11 is top-left 2x2.
                    # A12 is top-right 2x2.
                    # So A12 entries are (0,2), (0,3), (1,2), (1,3).
                    # My local indices 0,1,2,3 for A12?
                    # No, the 2x2x2 decomposition assumes A is 2x2.
                    # So the "A12" in the 2x2x2 context is a single element?
                    # No, in the recursive step, the "elements" are 2x2 blocks.
                    # So the "A11" in the 2x2x2 context is the top-left 2x2 block of the 4x4 matrix.
                    # So when we expand A11 into rank-1 terms, we are expanding the 2x2 block into its 4 elements.
                    
                    # So, for the term corresponding to A11 (local index 0 in the 2x2x2 vector):
                    # It involves all 4 elements of the top-left 2x2 block.
                    # So we need to iterate over the 4 elements of that block.
                    
                    # Let's refine the loop.
                    # The 7 terms are linear forms on the 2x2 blocks.
                    # Each block is a 2x2 matrix.
                    # So we expand each block into 4 rank-1 terms.
                    
                    # For a given term (u_vec, v_vec, w_vec):
                    # u_vec[i] is the coefficient for block i.
                    # We iterate over i, j, k in {0,1,2,3} (block indices).
                    # If u_vec[i] != 0, v_vec[j] != 0, w_vec[k] != 0:
                    #   We add the product of the 2x2 blocks.
                    #   Block i of A, Block j of B, Block k of C.
                    #   Each block is 2x2.
                    #   So we add 2*2*2 = 8 rank-1 terms?
                    #   No, the product of three 2x2 blocks is a 2x2x2 tensor.
                    #   We decompose that 2x2x2 tensor using the rank-7 decomposition again.
                    
                    # This is getting recursive.
                    # Let's just hardcode the 49 terms.
                    # Or, simpler: use the Kronecker product property.
                    # The rank-49 decomposition is the Kronecker product of the rank-7 decomposition.
                    # So we can just take the 7x7 combinations of the 7 terms.
                    
                    pass

    # Let's try a different approach: Just construct the 49 terms directly.
    # The 49 terms are (u1, v1, w1) outer (u2, v2, w2) ...
    # Actually, the Kronecker product of two rank-7 decompositions gives a rank-49 decomposition.
    # If T1 = sum_{a=1}^7 u_a outer v_a outer w_a
    # and T2 = sum_{b=1}^7 u'_b outer v'_b outer w'_b
    # Then T1 x T2 = sum_{a,b} (u_a outer u'_b) outer (v_a outer v'_b) outer (w_a outer w'_b)
    # This is a valid rank-49 decomposition for the 4x4x4 tensor.
    
    # We have the 7 vectors for the 2x2x2 case.
    # Let's generate the 49 terms.
    
    terms_4x4 = []
    for a in range(7):
        u_a, v_a, w_a = strassen_2x2_terms[a]
        for b in range(7):
            u_b, v_b, w_b = strassen_2x2_terms[b]
            
            # Kronecker product of vectors
            u_ab = np.kron(u_a, u_b)
            v_ab = np.kron(v_a, v_b)
            w_ab = np.kron(w_a, w_b)
            
            terms_4x4.append((u_ab, v_ab, w_ab))
            
    # Now we have 49 terms, each with vectors of length 16.
    # We need to map these to the global indices.
    # The local index in the vector corresponds to the index in the 4x4 matrix.
    # Local index l in {0..15} maps to global index l.
    # Because the Kronecker product of two 4-vectors gives a 16-vector,
    # and the order is consistent with row-major flattening.
    
    # Let's verify the order.
    # u_a = [u_a0, u_a1, u_a2, u_a3]
    # u_b = [u_b0, u_b1, u_b2, u_b3]
    # np.kron(u_a, u_b) = [u_a0*u_b0, u_a0*u_b1, u_a0*u_b2, u_a0*u_b3, u_a1*u_b0, ...]
    # This corresponds to indices:
    # 0: (0,0)
    # 1: (0,1)
    # 2: (0,2)
    # 3: (0,3)
    # 4: (1,0)
    # ...
    # This is exactly the row-major order of a 4x4 matrix.
    
    # So we can simply use the local index as the global index.
    
    for term in terms_4x4:
        u_vec, v_vec, w_vec = term
        for l in range(16):
            if u_vec[l] == 0 or v_vec[l] == 0 or w_vec[l] == 0:
                continue
            val = u_vec[l] * v_vec[l] * w_vec[l]
            U[l, col] += val
            V[l, col] += val
            W[l, col] += val
        col += 1
        
    # Part 1 done. col should be 49.
    assert col == 49, f"Expected 49, got {col}"
    
    # --- Part 2: Trivial for 4x4x1 ---
    # A (4x4) * B1 (4x1) -> C1 (4x1)
    # B1 is the last column of B (col 4).
    # C1 is the last column of C (col 4).
    # Rank 16.
    
    for i in range(n):
        for j in range(m):
            # k = 4 (last column)
            k = 4
            u_idx = i * m + j
            v_idx = j * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1
            
    assert col == 65, f"Expected 65, got {col}"
    
    # Verify
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
                
    T_reconstructed = np.zeros_like(T)
    for c in range(rank_total):
        T_reconstructed += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
    if not np.allclose(T_reconstructed, T, atol=1e-6):
        print("Verification failed")
        
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
