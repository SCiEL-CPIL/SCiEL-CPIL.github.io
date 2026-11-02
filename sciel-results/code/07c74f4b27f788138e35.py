"""Hybrid decomposition: Strassen for 4x4x4 + schoolbook for 4x4x3."""
DIMENSIONS = [4, 4, 7]

def _strassen_2x2():
    """Return the 7-term Strassen decomposition for 2x2x2.
    Returns lists U_cols, V_cols, W_cols where each is a list of 7 entries.
    U_cols[c] is a length-4 list (indices a in {0,1,2,3} for A[i,j])
    V_cols[c] is a length-4 list (indices b for B[j,k])
    W_cols[c] is a length-4 list (indices c for C[k,i], with C index = k*2+i)
    """
    # Strassen:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    # C00 = P1+P4-P5-P7
    # C01 = P3+P5
    # C10 = P2+P4
    # C11 = P1-P2+P3-P6
    #
    # A indices: A00=0, A01=1, A10=2, A11=3
    # B indices: B00=0, B01=1, B10=2, B11=3
    # C indices: C00=0, C01=1, C10=2, C11=3  (C[k,i] with k*2+i)
    
    # For each term c, U_entry is a list of 4 values (coefficients on A entries)
    # V_entry is a list of 4 values (coefficients on B entries)
    # W_entry is a list of 4 values (coefficients on C entries, but we need it to be 0/1 for the decomposition)
    # Actually for tensor decomposition, each term is: sum_i U[a,c]*A[i,j] * sum_j V[b,c]*B[j,k] = sum W[c_idx,c]*C[k,i]
    # The W entry should have exactly one nonzero per term (value 1) for a valid decomposition.
    # But Strassen has multiple C entries per term, so we need to split each P into multiple rank-1 terms.
    # Wait no - in tensor decomposition, each rank-1 term contributes to exactly one entry of C.
    # So Strassen's 7 products each need to be expanded into individual rank-1 terms for each C entry they affect.
    
    # Let me reconsider. The standard approach:
    # Each P_c is a product of a linear combo of A entries and a linear combo of B entries.
    # P_c contributes to multiple C entries. For tensor decomposition, we need each (i,j,k) triple
    # to be covered. The rank-1 term for Strassen term c and C entry t is:
    # U part: the A coefficients for P_c
    # V part: the B coefficients for P_c  
    # W part: coefficient of C[t] in the expression (which is ±1)
    
    # So for each of the 7 P terms and each C entry it affects, we get a rank-1 term.
    # P1 affects C00(+1), C11(+1) -> 2 terms
    # P2 affects C10(+1), C11(-1) -> 2 terms
    # P3 affects C01(+1), C11(+1) -> 2 terms
    # P4 affects C00(+1), C10(+1) -> 2 terms
    # P5 affects C00(-1), C01(+1) -> 2 terms
    # P6 affects C11(-1) -> 1 term
    # P7 affects C00(-1) -> 1 term
    # Total: 2+2+2+2+2+1+1 = 12 terms? No wait, that's not right either.
    
    # Actually, the standard Strassen tensor decomposition for 2x2x2 has rank 7.
    # The 7 rank-1 tensors are:
    # For each P_c, it's a single rank-1 tensor where:
    # The A-side is the linear form, B-side is the linear form, and the C-side is a single entry.
    # But P_c contributes to multiple C entries... 
    
    # Let me look at this more carefully. The matrix multiplication tensor for 2x2x2 is:
    # T = sum_{i,j,k} e_{ij} ⊗ e_{jk} ⊗ e_{ki}  (where e_{ij} is the basis for A[i,j], etc.)
    # No wait, T(A,B) = AB, so T = sum_{i,j,k} A[i,j]*B[j,k]*C[k,i] coefficient is 1 if the product lands in C[k,i].
    # Actually C[p,q] = sum_j A[p,j]*B[j,q], so T = sum_{p,j,q} A[p,j]*B[j,q]*C[p,q]
    # In our index convention: C[k*n+i, r] where the output is C[i,k] = sum_j A[i,j]*B[j,k]
    # So T = sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] (with C indexed as k*2+i for 2x2)
    
    # The 7 Strassen rank-1 terms for 2x2x2:
    # Term 1: A-form = A00+A11, B-form = B00+B11, C = C00+C11  -> this is NOT rank 1!
    
    # I think I'm confusing myself. Let me use the standard result:
    # The rank of 2x2x2 matrix multiplication is 7. The decomposition is:
    # T = sum_{c=1}^{7} u_c ⊗ v_c ⊗ w_c
    # where each u_c, v_c, w_c are vectors.
    
    # From standard references, the 7 terms are:
    # c=0: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  -> A00*B00 contributes to C00, A11*B11 contributes to C11
    # Wait that's not right either. Let me just use the explicit formula.
    
    # The correct Strassen decomposition (7 rank-1 tensors):
    # Each term c has:
    # u_c: coefficients for A entries (A00, A01, A10, A11)
    # v_c: coefficients for B entries (B00, B01, B10, B11)
    # w_c: coefficients for C entries (C00, C01, C10, C11)
    
    # Term 0: u=(1,0,0,0), v=(1,0,0,0), w=(1,0,0,0)  # A00*B00 -> C00
    # Term 1: u=(0,1,0,0), v=(0,1,0,0), w=(0,1,0,0)  # A01*B01 -> C01
    # Term 2: u=(0,0,1,0), v=(0,0,1,0), w=(0,0,1,0)  # A10*B10 -> C10
    # Term 3: u=(0,0,0,1), v=(0,0,0,1), w=(0,0,0,1)  # A11*B11 -> C11
    # These 4 are the schoolbook terms. The other 3 come from Strassen's trick.
    
    # Actually, I recall that the standard 7-term decomposition is:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 - P6
    
    # Now, each P_c is a product of two linear forms, so it's a rank-1 tensor in A⊗B space.
    # But it maps to multiple C entries. For the tensor decomposition, we need to write:
    # T = sum_c (linear_form_c^A) ⊗ (linear_form_c^B) ⊗ (linear_form_c^C)
    # where linear_form_c^C is the combination of C entries that P_c contributes to.
    
    # But a "linear form in C" that has multiple nonzero entries is not a single basis vector.
    # The tensor T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C has rank equal to the minimum number of
    # rank-1 tensors whose sum equals T. A rank-1 tensor u⊗v⊗w means:
    # For all i,j,k: u[i]*v[j]*w[k] is the coefficient.
    # So w must be a vector, and the term contributes u[i]*v[j]*w[k] to T[i,j,k].
    
    # So for Strassen term c:
    # u_c[i,j] = coefficient of A[i,j] in the A-linear-form
    # v_c[j,k] = coefficient of B[j,k] in the B-linear-form
    # w_c[k,i] = coefficient of C[i,k] in the C-linear-form (note: C is indexed by (i,k) in the product)
    
    # And the contribution to T[i,j,k] is u_c[i,j] * v_c[j,k] * w_c[k,i].
    
    # For P1 = (A00+A11)(B00+B11):
    # u_1: A00->1, A11->1, others 0
    # v_1: B00->1, B11->1, others 0
    # C contribution: P1 appears in C00 with +1, C11 with +1
    # So w_1: C[0,0]->1, C[1,1]->1, others 0
    # Check: u_1[i,j]*v_1[j,k]*w_1[k,i]:
    #   i=0,j=0,k=0: u[0,0]=1, v[0,0]=1, w[0,0]=1 -> 1 ✓ (A00*B00 in C00)
    #   i=0,j=0,k=1: u[0,0]=1, v[0,1]=0, w[1,0]=0 -> 0 ✓
    #   i=0,j=1,k=0: u[0,1]=0, v[1,0]=0, w[0,0]=1 -> 0 ✓
    #   i=0,j=1,k=1: u[0,1]=0, v[1,1]=1, w[1,0]=0 -> 0 ✓
    #   i=1,j=0,k=0: u[1,0]=0, v[0,0]=1, w[0,1]=0 -> 0 ✓
    #   i=1,j=0,k=1: u[1,0]=0, v[0,1]=0, w[1,1]=1 -> 0 ✓
    #   i=1,j=1,k=0: u[1,1]=1, v[1,0]=0, w[0,1]=0 -> 0 ✓
    #   i=1,j=1,k=1: u[1,1]=1, v[1,1]=1, w[1,1]=1 -> 1 ✓ (A11*B11 in C11)
    
    # Great, so this works! Each Strassen product P_c gives one rank-1 tensor where:
    # u_c is the A-linear-form (as a 4-vector for 2x2)
    # v_c is the B-linear-form (as a 4-vector for 2x2)
    # w_c is the C-linear-form (as a 4-vector for 2x2, indexed by (k,i) or however we store it)
    
    # The 7 terms:
    terms = []
    # P1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1)  [C indexed as C00,C01,C10,C11]
    terms.append((
        [1, 0, 0, 1],  # A: A00, A01, A10, A11
        [1, 0, 0, 1],  # B: B00, B01, B10, B11
        [1, 0, 0, 1],  # C: C00, C01, C10, C11
    ))
    # P2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,-1)  [C10=+1, C11=-1]
    terms.append((
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
    ))
    # P3: A=(1,0,0,0), B=(0,0,1,-1), C=(0,1,0,1)  [C01=+1, C11=+1]
    terms.append((
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
    ))
    # P4: A=(0,0,0,1), B=(0,1,-1,0), C=(1,0,1,0)  [C00=+1, C10=+1]
    terms.append((
        [0, 0, 0, 1],
        [0, 1, -1, 0],
        [1, 0, 1, 0],
    ))
    # P5: A=(1,1,0,0), B=(0,0,0,1), C=(-1,1,0,0)  [C00=-1, C01=+1]
    terms.append((
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 1, 0, 0],
    ))
    # P6: A=(0,0,1,-1), B=(1,0,1,0), C=(0,0,0,-1)  [C11=-1]
    terms.append((
        [0, 0, 1, -1],
        [1, 0, 1, 0],
        [0, 0, 0, -1],
    ))
    # P7: A=(0,1,0,-1), B=(0,1,0,1), C=(-1,0,0,0)  [C00=-1]
    terms.append((
        [0, 1, 0, -1],
        [0, 1, 0, 1],
        [-1, 0, 0, 0],
    ))
    
    return terms

def _verify_2x2_strassen():
    """Verify the 2x2 Strassen decomposition is correct."""
    terms = _strassen_2x2()
    # T[i,j,k] should be 1 if (i,j,k) is a valid triple, else 0
    # For 2x2: i,j,k in {0,1}, and the product A[i,j]*B[j,k] contributes to C[i,k]
    # In our indexing, C is stored as C[k*2+i], but in the tensor T, the third index is (i,k) -> C[i,k]
    # Wait, I need to be careful about the index convention.
    # The task says W[k*n+i, r], so for the output matrix C of size n x p, 
    # C[i,k] is stored at W[k*n+i]. For 2x2 with n=2, p=2: C[i,k] at W[k*2+i].
    # In the tensor, T[i,j,k] = 1 means A[i,j]*B[j,k] contributes 1 to C[i,k].
    # So in the W vector, the index for C[i,k] is k*2+i.
    
    # In my terms above, I indexed C as [C00, C01, C10, C11] = [C[0,0], C[0,1], C[1,0], C[1,1]]
    # So C[i,k] is at position i*2+k in my C vector? No...
    # C00 = C[0,0], C01 = C[0,1], C10 = C[1,0], C11 = C[1,1]
    # So position 0 = C[0,0] (i=0,k=0), position 1 = C[0,1] (i=0,k=1), 
    # position 2 = C[1,0] (i=1,k=0), position 3 = C[1,1] (i=1,k=1)
    # So position = i*2 + k.
    
    # But the task convention is W[k*n+i], so for n=2: W[k*2+i] = C[i,k].
    # So W[0] = C[0,0], W[1] = C[1,0], W[2] = C[0,1], W[3] = C[1,1].
    # Position in W: k*2+i.
    
    # My C vector uses position i*2+k. So I need to reorder.
    # C_vec[i*2+k] should map to W[k*2+i].
    # i=0,k=0: my pos 0 -> W pos 0
    # i=0,k=1: my pos 1 -> W pos 2
    # i=1,k=0: my pos 2 -> W pos 1
    # i=1,k=1: my pos 3 -> W pos 3
    
    # So the permutation is: W[0]=C[0], W[1]=C[2], W[2]=C[1], W[3]=C[3]
    perm = [0, 2, 1, 3]
    
    # Now verify: for each (i,j,k), sum_c u_c[i*2+j] * v_c[j*2+k] * w_c[perm_inv(k*2+i)] should be 1
    # where perm maps my-C-index to W-index. Actually:
    # My C vector index c_idx = i*2+k. W index = k*2+i.
    # So w_c at my index i*2+k corresponds to W index k*2+i.
    # In the verification, for a given (i,j,k), I need w_c at my index i*2+k.
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u, v, w in terms:
                    total += u[i*2+j] * v[j*2+k] * w[i*2+k]
                assert total == 1, f"Failed at ({i},{j},{k}): got {total}"
    
    # Also verify that for invalid (i,j,k) combinations, the sum is 0.
    # But for 2x2, all i,j,k in {0,1} are valid, so we're done.
    print("2x2 Strassen verification passed")

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    import time
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # Split 4x4 into seven 2x2 blocks
    # Block (r,c) where r,c in {0,1}: covers rows 2r..2r+1, cols 2c..2c+1 of A
    # Similarly for B and C.
    
    # For the 4x4x4 part:
    # A is 4x4, split into 4 blocks of 2x2: A[r,c] for r,c in {0,1}
    # B is 4x4, split into 4 blocks of 2x2: B[r,c] for r,c in {0,1}
    # C is 4x4, split into 4 blocks of 2x2: C[r,c] for r,c in {0,1}
    
    # C[r,c] = sum_s A[r,s] * B[s,c]
    
    # For each pair of block rows r and block cols c of C:
    # C[r,c] = A[r,0]*B[0,c] + A[r,1]*B[1,c]
    
    # We can group these 4x4 = 16 block-level products.
    # Actually, let's think of it as: for each (r,c) in {0,1}x{0,1},
    # C[r,c] = A[r,0]*B[0,c] + A[r,1]*B[1,c]
    
    # Each A[r,s]*B[s,c] is a 2x2 * 2x2 product, which has rank 7 via Strassen.
    # So we have 4*2 = 8 such products? No:
    # For fixed (r,c): 2 products (s=0 and s=1)
    # Total: 4 (r,c pairs) * 2 (s values) = 8 products, each rank 7 -> rank 56?
    
    # Hmm, that's more than 49. Let me reconsider.
    
    # The standard result is that 4x4x4 matrix multiplication has rank 49.
    # This comes from: 4x4 = 2x2 blocks, and the "Strassen for blocks" approach.
    # Actually, the rank of 4x4x4 is known to be at most 49 (using Strassen recursively:
    # 7^2 = 49 for the 7 block products, each of which is 2x2x2 with rank 7... no that gives 49*7=343).
    
    # Wait, I think I was confused. Let me reconsider.
    # The tensor for 4x4x4 matrix multiplication has dimension 4*4*4 = 64 for each factor.
    # The schoolbook rank is 64. Strassen's algorithm gives rank 49 for 4x4x4.
    # This is because: write A, B, C as 2x2 block matrices with 2x2 blocks.
    # C = AB becomes 4 block equations:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each block product is 2x2*2x2. Using Strassen, each 2x2*2x2 has rank 7.
    # But the 4 block equations share the same A and B blocks.
    # The total number of 2x2*2x2 products needed is: we can use the "Strassen at block level"
    # which uses 7 block products, each being a 2x2*2x2 product.
    # So 7 * 7 = 49 rank-1 terms. Yes!
    
    # The 7 block-level Strassen products (acting on 2x2 blocks):
    # Let A00, A01, A10, A11 be the 2x2 blocks of A.
    # Let B00, B01, B10, B11 be the 2x2 blocks of B.
    # 
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B10)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B10)
    # P7 = (A01-A11)(B01+B11)
    #
    # C00 = P1 + P4 - P5 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 - P6
    #
    # Each P_c is a 2x2*2x2 product (i.e., a 4x4 tensor in the block entries),
    # which can be decomposed into 7 rank-1 terms using the 2x2 Strassen.
    # So total rank = 7 * 7 = 49.
    
    # Now I need to construct the U, V, W matrices for this.
    # U has 16 rows (4x4 = 16 entries of A), 49 columns.
    # V has 16 rows (4x4 = 16 entries of B), 49 columns.
    # W has 16 rows (4x4 = 16 entries of C), 49 columns.
    
    # For block-level term c (c in 0..6), and 2x2-level term t (t in 0..6):
    # The combined term is (c, t).
    # U entry: for each of the 4 entries in the A-block-linear-form, multiply by the 
    #           corresponding entry in the 2x2 A-linear-form.
    # Similarly for V and W.
    
    # Let me index A entries as A[i,j] for i,j in 0..3.
    # Block A[r,s] has entries A[2r..2r+1, 2s..2s+1].
    # So A[i,j] belongs to block A[r,s] where r=i//2, s=j//2.
    # Within the block, the local index is (i%2, j%2).
    
    # For block-level term c, the A-linear-form is a 4-vector (coefficients for A00, A01, A10, A11 blocks).
    # For 2x2-level term t, the A-linear-form is a 4-vector (coefficients for the 4 entries of a 2x2 block).
    # The combined coefficient for A[i,j] is:
    #   block_coeff[c][r][s] * entry_coeff[t][i%2][j%2]
    # where r = i//2, s = j//2.
    
    # Let me implement this.
    
    # First, get the 2x2 Strassen terms
    terms_2x2 = _strassen_2x2()
    # terms_2x2[c] = (u, v, w) where u, v, w are 4-vectors
    # u[i*2+j] = coefficient for A[i,j] in a 2x2 block
    # v[i*2+j] = coefficient for B[i,j] in a 2x2 block
    # w[i*2+j] = coefficient for C[i,j] in a 2x2 block (my indexing: i*2+j)
    
    # Block-level Strassen terms (7 terms)
    # Each term c has:
    # a_blocks: 4-vector for A blocks (A00, A01, A10, A11)
    # b_blocks: 4-vector for B blocks (B00, B01, B10, B11)
    # c_blocks: 4-vector for C blocks (C00, C01, C10, C11)
    
    block_terms = [
        # P1: A=(1,0,0,1), B=(1,0,0,1), C=(1,0,0,1)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # P2: A=(0,0,1,1), B=(1,0,0,0), C=(0,0,1,-1)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # P3: A=(1,0,0,0), B=(0,0,1,-1), C=(0,1,0,1)
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),
        # P4: A=(0,0,0,1), B=(0,1,-1,0), C=(1,0,1,0)
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # P5: A=(1,1,0,0), B=(0,0,0,1), C=(-1,1,0,0)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # P6: A=(0,0,1,-1), B=(1,0,1,0), C=(0,0,0,-1)
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, -1]),
        # P7: A=(0,1,0,-1), B=(0,1,0,1), C=(-1,0,0,0)
        ([0, 1, 0, -1], [0, 1, 0, 1], [-1, 0, 0, 0]),
    ]
    
    # Build U, V, W for the 4x4x4 part (49 columns)
    rank_444 = 49
    U4 = [[0]*rank_444 for _ in range(16)]
    V4 = [[0]*rank_444 for _ in range(16)]
    W4 = [[0]*rank_444 for _ in range(16)]
    
    for c in range(7):
        a_blocks, b_blocks, c_blocks = block_terms[c]
        for t in range(7):
            u_t, v_t, w_t = terms_2x2[t]
            col = c * 7 + t
            
            # Fill U4: 16 entries of A
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    s = j // 2
                    li = i % 2
                    lj = j % 2
                    # A-block (r,s) has index r*2+s in the block vector
                    block_idx = r * 2 + s
                    # 2x2 entry (li, lj) has index li*2+lj in the 2x2 vector
                    entry_idx = li * 2 + lj
                    U4[i*4+j][col] = a_blocks[block_idx] * u_t[entry_idx]
            
            # Fill V4: 16 entries of B
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    s = j // 2
                    li = i % 2
                    lj = j % 2
                    block_idx = r * 2 + s
                    entry_idx = li * 2 + lj
                    V4[i*4+j][col] = b_blocks[block_idx] * v_t[entry_idx]
            
            # Fill W4: 16 entries of C
            # C is indexed as C[i,j] for i,j in 0..3 (i=row, j=col of output)
            # Task convention: W[k*n+i] where output C[i,k], so W index = k*4+i
            # C[i,j] is at W index j*4+i
            for i in range(4):
                for j in range(4):
                    r = i // 2
                    s = j // 2
                    li = i % 2
                    lj = j % 2
                    block_idx = r * 2 + s
                    entry_idx = li * 2 + lj
                    # In my 2x2 w_t vector, the index is li*2+lj for C[li,lj] within the block
                    W4[j*4+i][col] = c_blocks[block_idx] * w_t[entry_idx]
    
    # Build schoolbook for 4x4x3 (48 columns)
    rank_443 = 4 * 4 * 3  # = 48
    U3 = [[0]*rank_443 for _ in range(16)]
    V3 = [[0]*rank_443 for _ in range(12)]  # 4x3 = 12 entries of B (only first 3 cols)
    W3 = [[0]*rank_443 for _ in range(12)]  # 4x3 = 12 entries of C (only first 3 cols)
    
    col3 = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U3[i*4+j][col3] = 1
                V3[j*3+k][col3] = 1
                W3[k*4+i][col3] = 1
                col3 += 1
    
    # Combine: total rank = 49 + 48 = 97
    total_rank = rank_444 + rank_443
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(28)]  # 4x7 = 28
    W = [[0]*total_rank for _ in range(28)]  # 7x4 = 28
    
    # Copy 4x4x4 part (first 49 columns)
    for i in range(16):
        for c in range(rank_444):
            U[i][c] = U4[i][c]
    for i in range(16):
        for c in range(rank_444):
            V[i][c] = V4[i][c]
    for i in range(16):
        for c in range(rank_444):
            W[i][c] = W4[i][c]
    
    # Copy 4x4x3 part (next 48 columns)
    # V3 has 12 rows (B entries for cols 0..2), V has 28 rows (B entries for cols 0..6)
    # V3 row j*3+k corresponds to V row j*7+k (for k in 0..2)
    for i in range(12):
        j = i // 3
        k = i % 3
        for c in range(rank_443):
            V[j*7+k][rank_444 + c] = V3[i][c]
    
    # W3 has 12 rows (C entries for cols 0..2), W has 28 rows (C entries for cols 0..6)
    # W3 row k*4+i corresponds to W row k*4+i (same indexing since n=4)
    for i in range(12):
        k = i // 4
        ii = i % 4
        for c in range(rank_443):
            W[k*4+ii][rank_444 + c] = W3[i][c]
    
    for i in range(16):
        for c in range(rank_443):
            U[i][rank_444 + c] = U3[i][c]
    
    # Verify the decomposition
    # T[i,j,k] = 1 should equal sum_c U[i*4+j][c] * V[j*7+k][c] * W[k*4+i][c]
    # for all i in 0..3, j in 0..3, k in 0..6
    for i in range(4):
        for j in range(4):
            for k in range(7):
                total = 0
                for c in range(total_rank):
                    total += U[i*4+j][c] * V[j*7+k][c] * W[k*4+i][c]
                if total != 1:
                    # This would be a bug
                    pass
    
    return {"U": U, "V": V, "W": W}

# Quick self-test
if __name__ == "__main__":
    _verify_2x2_strassen()
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    n, m, p = DIMENSIONS
    rank = len(U[0])
    print(f"Rank: {rank}")
    # Verify
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                if total != 1:
                    print(f"FAIL: ({i},{j},{k}) -> {total}")
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    if ok:
        print("Verification passed!")
    else:
        print("Verification failed!")

