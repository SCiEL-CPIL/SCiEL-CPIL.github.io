import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build U, V, W as lists of lists.
    # We don't know the rank in advance, so we'll append.
    U = []
    V = []
    W = []
    
    def add_tensor(u_idx, v_idx, w_idx, coeff=1):
        # u_idx is index in A (i*m+j)
        # v_idx is index in B (j*p+k)
        # w_idx is index in C (k*n+i)
        # coeff is the multiplier (should be 1 for rank-1, but we can have -1 etc if we merge? 
        # No, rank-1 tensor is u v^T w^T. The coefficient is absorbed into U, V or W.
        # Since we need integer/half-integer entries, we can put the coeff in U.
        U.append(coeff)
        V.append(1)
        W.append(1)
        # Wait, U, V, W are matrices of shape (dim, rank).
        # So we need to append to each row? No, we are building column by column.
        # It is better to build as list of columns.
        pass

    # Better structure:
    # cols_U = []
    # cols_V = []
    # cols_W = []
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_rank1(u_vec, v_vec, w_vec, scale=1):
        # u_vec, v_vec, w_vec are dicts {index: coeff} or sparse representations
        # For simplicity, let's use dense vectors of length n*m, m*p, p*n
        # But n*m=25, so dense is fine.
        u_dense = [0] * (n*m)
        v_dense = [0] * (m*p)
        w_dense = [0] * (p*n)
        
        for idx, val in u_vec.items():
            u_dense[idx] = val * scale
        for idx, val in v_vec.items():
            v_dense[idx] = val
        for idx, val in w_vec.items():
            w_dense[idx] = val
            
        cols_U.append(u_dense)
        cols_V.append(v_dense)
        cols_W.append(w_dense)

    # Helper to get index in A (i*m+j)
    def idx_A(i, j): return i * m + j
    # Helper to get index in B (j*p+k)
    def idx_B(j, k): return j * p + k
    # Helper to get index in C (k*n+i)
    def idx_C(k, i): return k * n + i

    # 1. Generate 4x4 Strassen tensors
    # We need a function that generates the rank-1 tensors for n x n x n using Strassen.
    # It returns a list of (u_dict, v_dict, w_dict)
    
    def strassen_tensors(n, offset_i=0, offset_j=0, offset_k=0):
        # offset_i, offset_j, offset_k are the starting indices for this block in the global grid
        # This function assumes the block is n x n x n.
        # It returns a list of tuples (u_dict, v_dict, w_dict)
        
        if n == 1:
            # Single element
            i = offset_i
            j = offset_j
            k = offset_k
            u = {idx_A(i, j): 1}
            v = {idx_B(j, k): 1}
            w = {idx_C(k, i): 1}
            return [(u, v, w)]
        
        half = n // 2
        # Block indices:
        # A11: i in [0, half), j in [0, half)
        # A12: i in [0, half), j in [half, n)
        # A21: i in [half, n), j in [0, half)
        # A22: i in [half, n), j in [half, n)
        
        # B11: j in [0, half), k in [0, half)
        # B12: j in [0, half), k in [half, n)
        # B21: j in [half, n), k in [0, half)
        # B22: j in [half, n), k in [half, n)
        
        # C11: k in [0, half), i in [0, half)
        # C12: k in [0, half), i in [half, n)
        # C21: k in [half, n), i in [0, half)
        # C22: k in [half, n), i in [half, n)
        
        tensors = []
        
        # M1 = (A11+A22)(B11+B22)
        # We need to combine the tensors for A11 and A22 into one U vector, etc.
        # But Strassen works by forming the linear combinations FIRST, then multiplying.
        # So for M1, the U vector is the sum of U vectors for A11 and A22.
        # The V vector is the sum of V vectors for B11 and B22.
        # The W vector is... well, the result is added to C11 and C22.
        # So we need to add the resulting rank-1 tensors to the appropriate C blocks.
        
        # Let's get the base tensors for the sub-blocks
        # Tensors for A11 * B11 (contributes to C11)
        # This is getting complicated to manage the "linear combination" step directly in the tensor list.
        
        # Alternative:
        # Recursively generate the tensors for the 7 multiplications.
        # Each multiplication is between two n/2 x n/2 matrices.
        # Let's define the 7 pairs of (U_source, V_source, W_target_blocks)
        
        # U_source is a dict of indices in A
        # V_source is a dict of indices in B
        # W_target is a dict of indices in C with coefficients
        
        # Step 1: Get the "basis" tensors for the sub-blocks?
        # No, we just need to construct the U and V vectors for the 7 products.
        
        # Let's define the 7 products explicitly for n=2 (half=1)
        # But we are generalizing to n.
        
        # For M1:
        # U_M1 = U(A11) + U(A22)  <-- This is not a single tensor, it's a sum of tensors.
        # Wait, the rank-1 decomposition requires each term to be a single rank-1 tensor.
        # Strassen's algorithm says:
        # M1 = (A11+A22)(B11+B22)
        # This is a product of two matrices.
        # If we decompose A11+A22 into rank-1 terms, and B11+B22 into rank-1 terms,
        # the product is a sum of rank-1 terms.
        # So we need to recursively decompose the linear combinations.
        
        # This implies that `strassen_tensors` should return the list of rank-1 tensors 
        # that compute the matrix product for the given block.
        
        # If n=1, it returns 1 tensor.
        # If n=2, it returns 7 tensors.
        # If n=4, it returns 49 tensors.
        
        # How to construct the n=2 case from n=1 cases?
        # We need to form the linear combinations of the n=1 tensors.
        
        # Let's define a helper to add two sparse vectors (dicts).
        def add_dicts(d1, d2):
            res = {}
            for k, v in d1.items(): res[k] = res.get(k, 0) + v
            for k, v in d2.items(): res[k] = res.get(k, 0) + v
            return res

        def sub_dicts(d1, d2):
            res = {}
            for k, v in d1.items(): res[k] = res.get(k, 0) + v
            for k, v in d2.items(): res[k] = res.get(k, 0) - v
            return res

        # We need the U, V, W vectors for the sub-blocks.
        # But the sub-blocks are themselves products.
        # This recursive structure is tricky because the "inputs" to the Strassen steps 
        # are linear combinations of the original inputs, not the results of sub-products.
        
        # Correct Approach:
        # The rank-1 tensors for M_{n,n,n} are formed by the outer products of the 
        # linear forms in A and B.
        # For n=2, the 7 linear forms in A are:
        # A1 = A11 + A22
        # A2 = A21 + A22
        # A3 = A11
        # A4 = A22
        # A5 = A11 + A12
        # A6 = A21 - A11
        # A7 = A12 - A22
        
        # And in B:
        # B1 = B11 + B22
        # B2 = B11
        # B3 = B21 - B22
        # B4 = B12 - B11
        # B5 = B22
        # B6 = B11 + B12
        # B7 = B21 + B22
        
        # Each product $A_k B_k$ is a matrix product of size 1x1? No, they are scalars?
        # No, for n=2, A11 is a 1x1 matrix (scalar).
        # So $A_1 B_1$ is a scalar multiplication.
        # This scalar is then added to C11, C22, etc.
        
        # So for n=2, we have 7 scalar multiplications.
        # Each scalar multiplication corresponds to 1 rank-1 tensor.
        # The U vector has 1 at the position of the element in A1 (which is a sum of elements).
        # So U vector has non-zero entries at A11 and A22.
        
        # So, for n=2, we can explicitly define the 7 U, V, W vectors.
        
        # For n=4, we apply Strassen to the 2x2 blocks.
        # The "scalars" in the n=2 Strassen step are actually 2x2 matrices.
        # So $A_1 B_1$ is a 2x2 matrix product.
        # We decompose this 2x2 product using Strassen (7 rank-1 tensors).
        # So each of the 7 "big" terms expands to 7 "small" rank-1 tensors.
        
        # So the algorithm is:
        # 1. Define the 7 linear forms for U (in terms of block indices) and V.
        # 2. For each pair (U_form, V_form), recursively generate the tensors for the sub-block product.
        # 3. The W target for the sub-block product is determined by which C block it contributes to.
        
        # This requires knowing which C blocks each M_k contributes to.
        
        # M1 -> C11, C22
        # M2 -> C21, C22
        # M3 -> C11, C12
        # M4 -> C11, C21
        # M5 -> C11, C12
        # M6 -> C22
        # M7 -> C11, C22
        
        # Wait, standard Strassen:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So:
        # M1 contributes to C11 and C22.
        # M2 contributes to C21 and C22.
        # M3 contributes to C11 and C12.
        # M4 contributes to C11 and C21.
        # M5 contributes to C11 and C12.
        # M6 contributes to C22.
        # M7 contributes to C11 and C22.
        
        # When we recursively generate the tensors for the product $A_k B_k$ (which is a sub-block product),
        # we need to add the resulting W vectors to the appropriate C blocks.
        
        # This is implementable.
        
        # Let's implement `strassen_tensors(n, off_i, off_j, off_k)`.
        
        if n == 1:
            i = off_i
            j = off_j
            k = off_k
            u = {idx_A(i, j): 1}
            v = {idx_B(j, k): 1}
            w = {idx_C(k, i): 1}
            return [(u, v, w)]
            
        half = n // 2
        
        # Define the 7 U forms (as dicts of block offsets)
        # Block 0: (0,0), Block 1: (0,1), Block 2: (1,0), Block 3: (1,1)
        # Indices in A: (bi, bj)
        
        # U1: A11 + A22 -> (0,0) + (1,1)
        # U2: A21 + A22 -> (1,0) + (1,1)
        # U3: A11 -> (0,0)
        # U4: A22 -> (1,1)
        # U5: A11 + A12 -> (0,0) + (0,1)
        # U6: A21 - A11 -> (1,0) - (0,0)
        # U7: A12 - A22 -> (0,1) - (1,1)
        
        U_forms = [
            { (0,0): 1, (1,1): 1 },
            { (1,0): 1, (1,1): 1 },
            { (0,0): 1 },
            { (1,1): 1 },
            { (0,0): 1, (0,1): 1 },
            { (1,0): 1, (0,0): -1 },
            { (0,1): 1, (1,1): -1 }
        ]
        
        # V forms
        # B1: B11 + B22 -> (0,0) + (1,1)
        # B2: B11 -> (0,0)
        # B3: B21 - B22 -> (1,0) - (1,1)
        # B4: B12 - B11 -> (0,1) - (0,0)
        # B5: B22 -> (1,1)
        # B6: B11 + B12 -> (0,0) + (0,1)
        # B7: B21 + B22 -> (1,0) + (1,1)
        
        V_forms = [
            { (0,0): 1, (1,1): 1 },
            { (0,0): 1 },
            { (1,0): 1, (1,1): -1 },
            { (0,1): 1, (0,0): -1 },
            { (1,1): 1 },
            { (0,0): 1, (0,1): 1 },
            { (1,0): 1, (1,1): 1 }
        ]
        
        # W targets (coefficients for C blocks)
        # C blocks: (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
        W_targets = [
            { (0,0): 1, (1,1): 1 },   # M1
            { (1,0): 1, (1,1): 1 },   # M2
            { (0,0): 1, (0,1): 1 },   # M3
            { (0,0): 1, (1,0): 1 },   # M4
            { (0,0): 1, (0,1): 1 },   # M5
            { (1,1): 1 },             # M6
            { (0,0): 1, (1,1): 1 }    # M7
        ]
        
        all_tensors = []
        
        for t in range(7):
            u_form = U_forms[t]
            v_form = V_forms[t]
            w_target = W_targets[t]
            
            # We need to expand the product of the linear forms in u_form and v_form.
            # This is a product of two matrices of size half x half.
            # We recursively get the tensors for this sub-product.
            
            # The sub-product is (sum u_bi A_bi) * (sum v_bj B_bj)
            # = sum_{bi, bj} u_bi v_bj (A_bi * B_bj)
            
            # We iterate over all pairs of blocks (bi, bj)
            for (bi, bj), u_coeff in u_form.items():
                for (bj2, bk), v_coeff in v_form.items():
                    # Note: v_form keys are (block_row, block_col) in B.
                    # B block (bj2, bk) corresponds to A block row bi?
                    # No, the multiplication is A_bi * B_bj.
                    # A_bi is at row block bi, col block bj?
                    # Wait, A is n x n. B is n x n.
                    # A block (bi, bj) multiplies B block (bj, bk).
                    # The result goes to C block (bi, bk).
                    
                    # In our U_form, the key is (bi, bj)?
                    # My U_form definition above:
                    # { (0,0): 1, (1,1): 1 } means A11 and A22.
                    # A11 is row block 0, col block 0.
                    # A22 is row block 1, col block 1.
                    # So the key is (row_block, col_block).
                    
                    # In V_form:
                    # { (0,0): 1, (1,1): 1 } means B11 and B22.
                    # B11 is row block 0, col block 0.
                    # B22 is row block 1, col block 1.
                    
                    # So for a term u_coeff * A_bi_bj * v_coeff * B_bj_bk:
                    # The A block is (bi, bj).
                    # The B block is (bj, bk).
                    # The C block is (bi, bk).
                    
                    # We need to get the tensors for the product of A_bi_bj and B_bj_bk.
                    # These are half x half matrices.
                    
                    # Offsets:
                    # A offset: (off_i + bi*half, off_j + bj*half)
                    # B offset: (off_j + bj*half, off_k + bk*half)
                    # C offset: (off_k + bk*half, off_i + bi*half)
                    
                    # Recursive call for the sub-product
                    sub_tensors = strassen_tensors(half, 
                                                  off_i + bi*half, 
                                                  off_j + bj*half, 
                                                  off_k + bk*half)
                    
                    # Scale the sub-tensors by u_coeff * v_coeff
                    # And add to the appropriate W target blocks
                    
                    for (su, sv, sw) in sub_tensors:
                        # su, sv, sw are dicts
                        # We need to add sw to the global W vector, 
                        # but scaled by w_target coefficients.
                        
                        # Wait, the sub_tensors compute the product A_sub * B_sub.
                        # This product contributes to the C_sub block.
                        # But in the Strassen step, this product is added to 
                        # potentially multiple C blocks (as per w_target).
                        
                        # So we need to create a new W vector for the global tensor
                        # that is the sum of the sub W vectors, scaled by w_target.
                        
                        new_u = {}
                        for idx, val in su.items():
                            new_u[idx] = new_u.get(idx, 0) + val * u_coeff * v_coeff
                        
                        new_v = {}
                        for idx, val in sv.items():
                            new_v[idx] = new_v.get(idx, 0) + val * u_coeff * v_coeff # Wait, V is not scaled by u_coeff?
                        # Actually, the tensor is u v^T w^T.
                        # The coefficient u_coeff * v_coeff can be put in U or V.
                        # Let's put it in U.
                        
                        # Re-do scaling:
                        new_u = {}
                        for idx, val in su.items():
                            new_u[idx] = new_u.get(idx, 0) + val * u_coeff * v_coeff
                            
                        new_v = {}
                        for idx, val in sv.items():
                            new_v[idx] = new_v.get(idx, 0) + val
                            
                        new_w = {}
                        # The sub W vector sw corresponds to the C block (bi, bk).
                        # But the global W vector needs to add this to all C blocks 
                        # specified in w_target.
                        # Wait, the sub_tensors' W vector sw already has indices 
                        # corresponding to the sub C block (bi, bk) in the local coordinate system?
                        # No, `strassen_tensors` returns W vectors with global indices?
                        # My `strassen_tensors` uses `idx_C(k, i)` which uses global n, m, p.
                        # So `sw` already has global indices for the C block (bi, bk).
                        
                        # So we just need to add `sw` to the global W vector,
                        # but multiplied by the w_target coefficient for that block?
                        # No, the w_target says which C blocks this M_k contributes to.
                        # But the sub-product A_bi_bj * B_bj_bk ONLY produces the C_bi_bk block.
                        # So if w_target has non-zero entries for other blocks, 
                        # it means the *same* product M_k is added to those blocks.
                        
                        # So we need to create a W vector that has the entries of `sw`
                        # but also copies of them to the other target blocks?
                        # No, the indices in `sw` are specific to the (bi, bk) block.
                        # If w_target says M1 contributes to C11 and C22,
                        # and the sub-product is for C11 (bi=0, bk=0),
                        # then we add `sw` to C11.
                        # But M1 is *also* added to C22.
                        # Does the sub-product for C11 contribute to C22?
                        # No. The value of M1 is the same.
                        # So we need to add the *same* values to C22 as well.
                        
                        # This means we need to map the indices in `sw` (which are in C11)
                        # to the corresponding indices in C22?
                        # No, the indices in C are (k, i).
                        # C11 is k in [0, half), i in [0, half).
                        # C22 is k in [half, n), i in [half, n).
                        # The values are the same, but the indices are different.
                        
                        # So we need to shift the indices.
                        
                        # This is getting very complex to implement correctly on the fly.
                        
                        pass

    # Given the complexity and time limit, I will stick to the 98-rank hybrid 
    # but I will NOT implement the full Strassen recursion for the 4x4 part 
    # if it's too risky. 
    # Instead, I will use the schoolbook method for the 4x4 part as well?
    # Rank 4x4 schoolbook is 64.
    # Total rank: 64 + 16 + 16 + 16 + 1 = 113.
    # 113 < 125.
    # This is a safe improvement.
    
    # Let's implement the 113-rank solution.
    
    cols_U = []
    cols_V = []
    cols_W = []
    
    def add_rank1_dense(u, v, w):
        cols_U.append(u)
        cols_V.append(v)
        cols_W.append(w)

    # 1. Core 4x4x4 (Schoolbook)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[idx_A(i, j)] = 1
                v[idx_B(j, k)] = 1
                w[idx_C(k, i)] = 1
                add_rank1_dense(u, v, w)
                
    # 2. Cross terms
    # 2.1 i in 0..3, j in 0..3, k=4
    for i in range(4):
        for j in range(4):
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            u[idx_A(i, j)] = 1
            v[idx_B(j, 4)] = 1
            w[idx_C(4, i)] = 1
            add_rank1_dense(u, v, w)
            
    # 2.2 i=4, j in 0..3, k in 0..3
    for j in range(4):
        for k in range(4):
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            u[idx_A(4, j)] = 1
            v[idx_B(j, k)] = 1
            w[idx_C(k, 4)] = 1
            add_rank1_dense(u, v, w)
            
    # 2.3 i in 0..3, j=4, k in 0..3
    for i in range(4):
        for k in range(4):
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            u[idx_A(i, 4)] = 1
            v[idx_B(4, k)] = 1
            w[idx_C(k, i)] = 1
            add_rank1_dense(u, v, w)
            
    # 2.4 i=4, j=4, k=4
    u = [0] * 25
    v = [0] * 25
    w = [0] * 25
    u[idx_A(4, 4)] = 1
    v[idx_B(4, 4)] = 1
    w[idx_C(4, 4)] = 1
    add_rank1_dense(u, v, w)
    
    # Convert to matrices
    rank = len(cols_U)
    U = [[0]*rank for _ in range(25)]
    V = [[0]*rank for _ in range(25)]
    W = [[0]*rank for _ in range(25)]
    
    for c in range(rank):
        for r in range(25):
            U[r][c] = cols_U[c][r]
            V[r][c] = cols_V[c][r]
            W[r][c] = cols_W[c][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
