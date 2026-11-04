"""Matrix multiplication 3x5x6: direct-sum decomposition via Strassen + Winograd-style blocks."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 5, 6

    # We decompose the 3x5x6 multiplication tensor as a direct sum:
    # Partition rows into a 1-row block and a 2-row block.
    # 
    # Block 1: 1x5x6 tensor (rank 10 via Winograd-style)
    # Block 2: 2x5x6 tensor (rank 84 via Kronecker: Strassen 2x2x2 rank-7 ⊗ 1x5x3 rank-12)
    # Total rank = 10 + 84 = 94
    #
    # The output index convention is W[k*n+i, r] where i is row of A (0..n-1),
    # k is column of B / row of C (0..p-1).
    #
    # For the direct sum, we partition the row index i into:
    #   i=0 -> Block 1 (1 row)
    #   i=1,2 -> Block 2 (2 rows)
    #
    # Within Block 2, we further partition the column index k (0..5) into pairs:
    #   (k=0,1), (k=2,3), (k=4,5) -> 3 pairs, each handled by a 2x2x2 Strassen block
    # But wait: Strassen handles 2x2x2. For 2x5x6, we need to partition differently.
    #
    # Actually, let me reconsider the decomposition more carefully.
    #
    # 2x5x6: We can write this as a sum over the middle dimension.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # For 2x5x6, one approach: partition the 5 columns of A (rows of B) into groups,
    # and the 6 rows of B (columns of C) into pairs.
    #
    # Actually, the strategy says: Kronecker product of rank-7 Strassen 2x2x2 and rank-12 1x5x3.
    # 2x5x6 = (2x2x2) ⊗ (1x5x3) in the tensor sense? That doesn't quite work dimensionally.
    #
    # Let me think differently. 2x5x6 can be seen as:
    # For each pair of output columns (k, k+1) for k in {0,2,4}, we have a 2x5x2 sub-problem.
    # 2x5x2: We can use a rank-? decomposition.
    #
    # Actually, let me just implement the schoolbook and try to find a better one.
    # Given the complexity, let me try a different approach:
    #
    # Use the fact that 3x5x6 can be decomposed by partitioning.
    # 
    # Let me try: partition the 3 rows into 1+2.
    # 1x5x6: rank at most 5*6=30, but we can do better.
    # 2x5x6: rank at most 2*5*6=60.
    #
    # For 1x5x6: C[k] = sum_j A[j]*B[j,k]. This is a linear combination.
    # We can use Winograd-like: partition j into groups of 3 and k into groups of 3.
    # 1x3x3 has rank 3 (it's a 3x3 matrix multiplication which is rank 9 schoolbook, 
    # but as a bilinear form 1x3x3, the tensor is just the 3x3 identity-like... 
    # Actually 1x3x3 tensor T[i,j,k] = delta_{i,0} * (something). 
    # Wait, 1x3x3 means A is 1x3, B is 3x3, C is 1x3.
    # C[k] = sum_j A[j] B[j,k]. This is just matrix-vector multiplication.
    # The tensor for this is T[j,k] (since i is fixed at 0), which is the identity 3x3.
    # Rank of identity 3x3 as a tensor? It's just 3 (it's a matrix, rank 3).
    # So 1x3x3 has rank 3.
    #
    # 1x5x6: partition j into {0,1,2} and {3,4}, k into {0,1,2} and {3,4,5}.
    # 1x3x3 (rank 3) for (j in 0..2, k in 0..2)
    # 1x2x3 for (j in 3..4, k in 0..2)
    # 1x3x3 (rank 3) for (j in 0..2, k in 3..5)
    # 1x2x3 for (j in 3..4, k in 3..5)
    #
    # 1x2x3: C[k] = A[0]*B[0,k] + A[1]*B[1,k]. Tensor is 2x3 matrix, rank 2.
    # So 1x5x6 rank <= 3 + 2 + 3 + 2 = 10. 
    #
    # For 2x5x6: Similarly partition.
    # 2x3x3: This is 2x3 matrix times 3x3 matrix = 2x3 matrix.
    # Tensor for 2x3x3: T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    # This is the tensor for multiplying a 2x3 by a 3x3. Rank at most 2*3*3=18 schoolbook.
    # Can we do better? 2x3x3 = (2x1x1) ⊗ (1x3x3)? No, that's not a Kronecker.
    # Actually 2x3x3 can be seen as 2 copies of 1x3x3, so rank 2*3=6? No, that's not right either.
    # The tensor T[i,j,k] for 2x3x3 matrix mult: C[i,k] = sum_j A[i,j] B[j,k].
    # This is a 2x3x3 tensor. Its rank... we can write it as sum over (i,k) of e_i ⊗ (row i of A pattern) ⊗ e_k.
    # Actually, it's just the Kronecker product structure: for each i, we have a 1x3x3 tensor.
    # So rank(2x3x3) <= 2 * rank(1x3x3) = 2*3 = 6? Let me verify.
    # C[i,k] = sum_j A[i,j] B[j,k]. For fixed i, this is 1x3x3 which has rank 3.
    # But the i index just selects which "slice" we're in, so the total tensor is the direct sum
    # of two 1x3x3 tensors, hence rank 6. Yes!
    #
    # So 2x5x6: partition j into {0,1,2} and {3,4}, k into {0,1,2} and {3,4,5}.
    # 2x3x3 (rank 6) for (j in 0..2, k in 0..2)
    # 2x2x3 for (j in 3..4, k in 0..2)
    # 2x3x3 (rank 6) for (j in 0..2, k in 3..5)
    # 2x2x3 for (j in 3..4, k in 3..5)
    #
    # 2x2x3: C[i,k] = sum_j A[i,j] B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}.
    # This is 2x2x3 tensor. Rank? For each k, we have a 2x2 matrix mult which has rank 4 (schoolbook) 
    # or 7 (Strassen). But as a single tensor 2x2x3, it's the direct sum over k of 2x2x1 tensors?
    # No. 2x2x3: T[i,j,k]. For fixed k, T[i,j,k] is a 2x2 matrix (the k-th column of B's contribution).
    # Actually T[i,j,k] = delta for the specific multiplication. The tensor is:
    # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k].
    # So T[i,j,k] is 1 for all (i,j,k) in the valid range? No, it's the structure tensor.
    # For matrix multiplication m1 x m2 x m3, the tensor T has T[i,j,k] = 1 where 
    # C[i,k] gets contribution from A[i,j]*B[j,k]. So T[i,j,k] = 1 for all valid i,j,k.
    # Wait no. The multiplication tensor for (n,m,p) is T in U*V*W where
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (the coefficient which is 1 for the standard mult).
    # Actually the structure tensor is T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^U ⊗ e_j^V ⊗ e_k^W
    # No that's not right either. Let me be precise.
    #
    # The matrix multiplication tensor T_{n,m,p} is defined by:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^B ⊗ e_k^C
    # where A is n*m dimensional (flattened), B is m*p, C is p*n.
    # And the contraction gives C[i,k] = sum_j A[i,j] B[j,k].
    #
    # So T_{n,m,p} has components T[a,b,c] where a indexes A (n*m), b indexes B (m*p), c indexes C (p*n).
    # T[a,b,c] = 1 if a=(i,j), b=(j',k), c=(k',i') and i=i', j=j', k=k'.
    # Otherwise 0.
    #
    # So for 2x2x3: a in {0,1,2,3} (i*2+j), b in {0,1,2,3,4,5} (j*3+k), c in {0,1,2,3,4,5} (k*2+i).
    # T[a,b,c] = 1 iff a=i*2+j, b=j*3+k, c=k*2+i for some i in {0,1}, j in {0,1}, k in {0,1,2}.
    #
    # The rank of this tensor: we can write it as sum over (i,j,k) of rank-1 terms.
    # That's 2*2*3=12 terms. Can we do better?
    # For fixed i, the sub-tensor is 1x2x3 which has rank 2 (as we established, it's a 2x3 matrix, rank 2).
    # So 2x2x3 is direct sum of two 1x2x3 tensors, rank 2*2=4.
    #
    # So 2x5x6 rank <= 6 + 4 + 6 + 4 = 20.
    #
    # Total: 1x5x6 (rank 10) + 2x5x6 (rank 20) = 30.
    #
    # That's much better than schoolbook 90!
    #
    # Let me implement this.

    def make_rank1_term(i, j, k, n, m, p):
        """Create a rank-1 term for the (i,j,k) entry of the multiplication tensor."""
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i
        return (a_idx, b_idx, c_idx)

    def build_direct_sum(n, m, p, rank_terms):
        """Build U, V, W from a list of rank-1 terms (a_idx, b_idx, c_idx)."""
        rank = len(rank_terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col, (a_idx, b_idx, c_idx) in enumerate(rank_terms):
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
        return U, V, W

    # Build 1x5x6 decomposition (rank 10)
    # Partition j: {0,1,2} and {3,4}; k: {0,1,2} and {3,4,5}
    # 1x3x3 (rank 3): j in 0..2, k in 0..2
    # 1x2x3 (rank 2): j in 3..4, k in 0..2
    # 1x3x3 (rank 3): j in 0..2, k in 3..5
    # 1x2x3 (rank 2): j in 3..4, k in 3..5

    def build_1x5x6(n, m, p):
        """Build rank-10 decomposition for 1x5x6 (row i=0)."""
        terms = []
        # 1x3x3 blocks: for j in 0..2, k in 0..2: rank-3 means 3 terms
        # 1x3x3 tensor: C[k] = sum_{j=0}^2 A[j] B[j,k]
        # Rank-3 decomposition: for each k in 0..2, one term with A[j] for all j, B[j,k] for all j
        # Actually, 1x3x3 has rank 3: T = sum_{k=0}^2 (sum_j e_j^A ⊗ e_{j,k}^B) ⊗ e_k^C
        # Wait, that's not rank-1. Let me think again.
        # 1x3x3: C[k] = A[0]B[0,k] + A[1]B[1,k] + A[2]B[2,k]
        # As a bilinear form, this is <A, B_k> where B_k is the k-th column of B.
        # The tensor T[j,k] (with i fixed) is the 3x3 identity matrix (T[j,k] = delta_{j,k}? No!)
        # T[j,k] = 1 for all j,k in 0..2? No.
        # C[k] = sum_j A[j] * B[j,k]. So the coefficient of A[j]*B[j,k] in C[k] is 1.
        # So T[j,k] = 1 for all j,k. That's the all-ones 3x3 matrix, which has rank 1!
        # Wait, that can't be right. Let me re-examine.
        #
        # The multiplication tensor for 1x3x3:
        # C[0] = A[0]B[0,0] + A[1]B[1,0] + A[2]B[2,0]
        # C[1] = A[0]B[0,1] + A[1]B[1,1] + A[2]B[2,1]
        # C[2] = A[0]B[0,2] + A[1]B[1,2] + A[2]B[2,2]
        #
        # So T[j, k] (for the 1x3x3 sub-tensor, with i=0 fixed) has T[j,k] = 1 for all j,k.
        # Because every A[j] multiplies every B[j,k] to contribute to C[k].
        # The tensor is T = sum_{j=0}^2 sum_{k=0}^2 e_j^A ⊗ e_{j,k}^B ⊗ e_k^C
        # = (sum_j e_j^A) ⊗ (sum_{j,k} e_{j,k}^B) ⊗ ... no that's not right.
        #
        # T = sum_{j,k} e_j^A ⊗ e_{j,k}^B ⊗ e_k^C
        # Can we factor this? T = (sum_j e_j^A ⊗ e_{j,*}^B) ... hmm.
        # Actually: T = sum_k (sum_j e_j^A ⊗ e_{j,k}^B) ⊗ e_k^C
        # Each (sum_j e_j^A ⊗ e_{j,k}^B) is a vector in A*space ⊗ B*space.
        # So T is a sum of 3 terms, each of the form v_k ⊗ e_k^C where v_k = sum_j e_j^A ⊗ e_{j,k}^B.
        # So rank <= 3. And indeed rank = 3 since the v_k are linearly independent.
        #
        # So for 1x3x3, the rank-3 decomposition is:
        # For k in 0..2: term with U having 1s at all j positions (for this block), 
        # V having 1s at all (j,k) positions, W having 1 at k position.
        #
        # In our indexing for the full 1x5x6 (i=0, m=5, p=6):
        # A index: i*m + j = 0*5 + j = j
        # B index: j*p + k = j*6 + k
        # C index: k*n + i = k*3 + 0 = k (since n=3, i=0)

        terms = []
        # 1x3x3 block 1: j in {0,1,2}, k in {0,1,2}
        for k in range(3):
            a_indices = [0, 1, 2]  # j values
            b_indices = [j * 6 + k for j in range(3)]  # (j, k) in B
            c_index = k * 3 + 0  # k*3 + i=0
            terms.append((a_indices, b_indices, c_index))
        
        # 1x2x3 block: j in {3,4}, k in {0,1,2}
        # C[k] = A[3]B[3,k] + A[4]B[4,k]
        # Rank 2: for k in 0..2... wait, 1x2x3 rank is 2?
        # T[j,k] = 1 for j in {0,1}, k in {0,1,2}. 
        # T = sum_{j,k} e_j ⊗ e_{j,k} ⊗ e_k = sum_k (sum_j e_j ⊗ e_{j,k}) ⊗ e_k
        # That's rank 3, not 2! I made an error.
        # 1x2x3: C[k] = A[0]B[0,k] + A[1]B[1,k] for k=0,1,2.
        # T[j,k] = 1 for all j in {0,1}, k in {0,1,2}.
        # T = sum_{j=0}^1 sum_{k=0}^2 e_j ⊗ e_{j,k} ⊗ e_k
        # = sum_{k=0}^2 (sum_{j=0}^1 e_j ⊗ e_{j,k}) ⊗ e_k
        # Rank = 3 (the three terms for k=0,1,2 are independent).
        #
        # Hmm, so 1x2x3 has rank 3, not 2. Let me reconsider.
        # Actually, can we do better? T[j,k] is a 2x3 all-ones matrix.
        # As a tensor T = sum_{j,k} e_j ⊗ f_{j,k} ⊗ g_k where f_{j,k} = e_{j,k} in B space.
        # The "matrix" view: for each k, we have a vector in A-space: [1,1]^T.
        # T = sum_k v ⊗ w_k where v = [1,1] in A-space, w_k = sum_j e_j ⊗ e_{j,k} in B-space.
        # The w_k for different k are linearly independent (they involve different B indices).
        # So rank = 3.
        #
        # Alternatively: T = sum_j u_j ⊗ z_j where u_j = sum_k e_k in C-space, z_j = e_j ⊗ (sum_k e_{j,k}) in B-space.
        # That gives rank 2! Let's check:
        # sum_j u_j ⊗ z_j = sum_j (sum_k e_k) ⊗ (e_j^A ⊗ sum_k e_{j,k}^B)
        # = sum_j sum_k e_k^C ⊗ e_j^A ⊗ e_{j,k}^B
        # = sum_{j,k} e_j^A ⊗ e_{j,k}^B ⊗ e_k^C = T. Yes!
        # So rank(1x2x3) = 2.
        #
        # Great, so my original estimate was correct. The rank-2 decomposition is:
        # For j in {0,1}: U has 1 at j, V has 1s at all (j,k) for k in range, W has 1s at all k.

        # 1x2x3 block: j in {3,4}, k in {0,1,2}
        for j in [3, 4]:
            a_indices = [j]
            b_indices = [j * 6 + k for k in range(3)]
            c_indices = [k * 3 + 0 for k in range(3)]
            terms.append((a_indices, b_indices, c_indices))
        
        # 1x3x3 block 2: j in {0,1,2}, k in {3,4,5}
        for k in range(3, 6):
            a_indices = [0, 1, 2]
            b_indices = [j * 6 + k for j in range(3)]
            c_index = k * 3 + 0
            terms.append((a_indices, b_indices, c_index))
        
        # 1x2x3 block: j in {3,4}, k in {3,4,5}
        for j in [3, 4]:
            a_indices = [j]
            b_indices = [j * 6 + k for k in range(3, 6)]
            c_indices = [k * 3 + 0 for k in range(3, 6)]
            terms.append((a_indices, b_indices, c_indices))
        
        return terms

    # Build 2x5x6 decomposition (rank 20)
    # Partition j: {0,1,2} and {3,4}; k: {0,1,2} and {3,4,5}
    # 2x3x3 (rank 6): i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 2x2x3 (rank 4): i in {0,1}, j in {3,4}, k in {0,1,2}
    # 2x3x3 (rank 6): i in {0,1}, j in {0,1,2}, k in {3,4,5}
    # 2x2x3 (rank 4): i in {0,1}, j in {3,4}, k in {3,4,5}

    def build_2x5x6(n, m, p):
        """Build rank-20 decomposition for 2x5x6 (rows i=1,2)."""
        terms = []
        # 2x3x3 blocks: for each i in {1,2}, it's a 1x3x3 (rank 3)
        # So 2x3x3 = direct sum of two 1x3x3, rank 6.
        for i in [1, 2]:
            for k in range(3):
                a_indices = [i * 5 + j for j in range(3)]
                b_indices = [j * 6 + k for j in range(3)]
                c_index = k * 3 + i
                terms.append((a_indices, b_indices, c_index))
        
        # 2x2x3: rank 4. For each i in {1,2}, 1x2x3 has rank 2.
        # 1x2x3 rank-2: for j in {0,1} (local), U has 1 at j, V has 1s at all (j,k), W has 1s at all k.
        for i in [1, 2]:
            for j_local in [0, 1]:
                j = 3 + j_local
                a_indices = [i * 5 + j]
                b_indices = [j * 6 + k for k in range(3)]
                c_indices = [k * 3 + i for k in range(3)]
                terms.append((a_indices, b_indices, c_indices))
        
        # 2x3x3 block 2: j in {0,1,2}, k in {3,4,5}
        for i in [1, 2]:
            for k in range(3, 6):
                a_indices = [i * 5 + j for j in range(3)]
                b_indices = [j * 6 + k for j in range(3)]
                c_index = k * 3 + i
                terms.append((a_indices, b_indices, c_index))
        
        # 2x2x3 block 2: j in {3,4}, k in {3,4,5}
        for i in [1, 2]:
            for j_local in [0, 1]:
                j = 3 + j_local
                a_indices = [i * 5 + j]
                b_indices = [j * 6 + k for k in range(3, 6)]
                c_indices = [k * 3 + i for k in range(3, 6)]
                terms.append((a_indices, b_indices, c_indices))
        
        return terms

    # Combine: 1x5x6 terms + 2x5x6 terms
    terms_1 = build_1x5x6(n, m, p)
    terms_2 = build_2x5x6(n, m, p)
    all_terms = terms_1 + terms_2
    
    rank = len(all_terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(all_terms):
        a_indices, b_indices, c_indices = term
        if isinstance(c_indices, int):
            c_indices = [c_indices]
        for a_idx in a_indices:
            U[a_idx][col] = 1
        for b_idx in b_indices:
            V[b_idx][col] = 1
        for c_idx in c_indices:
            W[c_idx][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
