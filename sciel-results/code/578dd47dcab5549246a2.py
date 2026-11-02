import time
import itertools

def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill

    n, m, p = 2, 5, 6
    # U: (n*m, rank) = (10, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (12, rank) with W[k*n + i, r]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    # Generate candidate rank-1 tensors
    # Each candidate is a tuple of (u_entries, v_entries, w_entries)
    # where each is a list of (index, value) pairs
    # The tensor equation: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = A[i,j]*B[j,k]
    # So for each (i,j,k), the product U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must sum to 1
    # (since A[i,j]*B[j,k] is the target, and we're decomposing the multiplication tensor)
    
    # Actually, the tensor is T[i,j,k] = delta (it's 1 for the multiplication)
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for each (i,j,k) in {0,1}x{0..4}x{0..5}, 
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # A rank-1 term is specified by three vectors u, v, w (each with one non-zero entry per relevant index)
    # But actually, each rank-1 term contributes to MULTIPLE (i,j,k) entries simultaneously.
    
    # Let me think differently. A rank-1 tensor is u ⊗ v ⊗ w where:
    # u is a vector in R^(n*m), v in R^(m*p), w in R^(p*n)
    # The (i,j,k) entry is u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    # For the multiplication tensor, T[i,j,k] = 1 for all valid (i,j,k)
    # Wait no - the multiplication tensor for (n,m,p) matrix mult has T[i,j,k] = 1 if the 
    # computation uses A[i,j] and B[j,k] to produce C[i,k]. For standard matrix mult,
    # C[i,k] = sum_j A[i,j] * B[j,k], so the tensor has T[i,j,k] = 1 for all i,j,k.
    
    # So we need: sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i] = 1 for all i in {0,1}, j in {0..4}, k in {0..5}
    
    # That's 2*5*6 = 60 equations.
    
    # A rank-1 term is defined by sparse vectors. Let's generate candidates:
    
    candidates = []
    
    # Schoolbook: for each (i,j,k), a term with u[i*m+j]=1, v[j*p+k]=1, w[k*n+i]=1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [(u_idx(i,j), 1)]
                v = [(v_idx(j,k), 1)]
                w = [(w_idx(k,i), 1)]
                candidates.append((u, v, w))
    
    # Strassen-like: for 2x2x2 blocks (i in {0,1}, j in {j1,j2}, k in {k1,k2})
    # 7 Strassen terms
    def strassen_terms(j1, j2, k1, k2):
        """Generate 7 Strassen rank-1 terms for 2x2x2 block."""
        terms = []
        # m1 = (a00+a11)(b00+b11)
        terms.append(
            ([(u_idx(0,j1),1), (u_idx(1,j2),1)],
             [(v_idx(j1,k1),1), (v_idx(j2,k2),1)],
             [(w_idx(k1,0),1), (w_idx(k2,1),1)])
        )
        # m2 = (a10+a11)*b00
        terms.append(
            ([(u_idx(1,j1),1), (u_idx(1,j2),1)],
             [(v_idx(j1,k1),1)],
             [(w_idx(k1,1),1), (w_idx(k2,1),-1)])
        )
        # m3 = a00*(b01-b11)
        terms.append(
            ([(u_idx(0,j1),1)],
             [(v_idx(j1,k2),1), (v_idx(j2,k2),-1)],
             [(w_idx(k2,0),1), (w_idx(k2,1),1)])
        )
        # m4 = a11*(b10-b00)
        terms.append(
            ([(u_idx(1,j2),1)],
             [(v_idx(j2,k1),1), (v_idx(j1,k1),-1)],
             [(w_idx(k1,0),1), (w_idx(k1,1),1)])
        )
        # m5 = (a00+a01)*b11
        terms.append(
            ([(u_idx(0,j1),1), (u_idx(0,j2),1)],
             [(v_idx(j2,k2),1)],
             [(w_idx(k1,0),-1), (w_idx(k2,0),1)])
        )
        # m6 = (a10-a00)*(b00+b01)
        terms.append(
            ([(u_idx(1,j1),1), (u_idx(0,j1),-1)],
             [(v_idx(j1,k1),1), (v_idx(j1,k2),1)],
             [(w_idx(k2,1),1)])
        )
        # m7 = (a01-a11)*(b10+b11)
        terms.append(
            ([(u_idx(0,j2),1), (u_idx(1,j2),-1)],
             [(v_idx(j2,k1),1), (v_idx(j2,k2),1)],
             [(w_idx(k1,0),1)])
        )
        return terms
    
    for j1 in range(m-1):
        for j2 in range(j1+1, m):
            for k1 in range(p-1):
                for k2 in range(k1+1, p):
                    candidates.extend(strassen_terms(j1, j2, k1, k2))
    
    # Also try 2x1x2 schoolbook blocks (i in {0,1}, single j, k in {k1,k2})
    # This is just 4 separate rank-1 terms, already covered by schoolbook above.
    
    # And 2x2x1 (i in {0,1}, j in {j1,j2}, single k)
    def strassen_2x2x1_terms(j1, j2, k):
        """For 2x2x1, we can use a simpler approach."""
        # C[0,k] = A[0,j1]*B[j1,k] + A[0,j2]*B[j2,k]
        # C[1,k] = A[1,j1]*B[j1,k] + A[1,j2]*B[j2,k]
        # Schoolbook: 4 terms
        terms = []
        for i in range(2):
            for j in [j1, j2]:
                terms.append(
                    ([(u_idx(i,j),1)],
                     [(v_idx(j,k),1)],
                     [(w_idx(k,i),1)])
                )
        return terms
    
    # These are already in schoolbook, so skip.
    
    # Now, each candidate has a "coverage" - which (i,j,k) entries it contributes to.
    # For a candidate (u, v, w), the contribution to entry (i,j,k) is:
    #   u_val[i*m+j] * v_val[j*p+k] * w_val[k*n+i]
    # where u_val, v_val, w_val are the full vectors (0 if not present).
    
    # Let's precompute for each candidate what it contributes to each of the 60 (i,j,k) entries.
    # An entry (i,j,k) is "covered" by a candidate if the candidate has non-zero values
    # at u_idx(i,j), v_idx(j,k), and w_idx(k,i).
    
    # Build a mapping: candidate_id -> dict from (i,j,k) to contribution value
    # But actually, for exact decomposition, we need the SUM of contributions = 1 for each (i,j,k).
    
    # Let me represent each candidate by its sparse contribution pattern.
    # For candidate c with u_entries, v_entries, w_entries:
    #   u_map = {idx: val for idx, val in u_entries}
    #   v_map = {idx: val for idx, val in v_entries}
    #   w_map = {idx: val for idx, val in w_entries}
    #   For each (i,j,k): contrib = u_map.get(i*m+j, 0) * v_map.get(j*p+k, 0) * w_map.get(k*n+i, 0)
    
    # Precompute all (i,j,k) entries
    all_entries = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For each candidate, compute its contribution vector (length 60)
    candidate_contribs = []
    for u, v, w in candidates:
        u_map = dict(u)
        v_map = dict(v)
        w_map = dict(w)
        contrib = []
        for (i, j, k) in all_entries:
            c = u_map.get(i*m+j, 0) * v_map.get(j*p+k, 0) * w_map.get(k*n+i, 0)
            contrib.append(c)
        candidate_contribs.append(contrib)
    
    # Now we need to select a subset of candidates such that for each entry e,
    # the sum of contributions = 1.
    
    # This is a linear system: sum_{c in S} contrib_c[e] = 1 for all e.
    
    # Greedy approach: at each step, pick the candidate that covers the most "unmet" entries
    # with the right sign, or use a more sophisticated approach.
    
    # Actually, let's try a different approach: since we know the structure, let's use
    # the block decomposition but optimize the number of Strassen vs schoolbook.
    
    # The 2x5x6 tensor can be decomposed as:
    # - 2x2x2 blocks: each needs rank 7 (Strassen) or rank 8 (schoolbook)
    # - 2x2x1 blocks: rank 4 (schoolbook) or maybe less?
    # - 2x1x2 blocks: rank 4 (schoolbook)
    # - 2x1x1 blocks: rank 2 (schoolbook)
    
    # For 2x2x1: C[0,k] = A[0,j1]*B[j1,k] + A[0,j2]*B[j2,k]
    #            C[1,k] = A[1,j1]*B[j1,k] + A[1,j2]*B[j2,k]
    # This is like multiplying two 2x1 vectors by a scalar (B[j,k]).
    # We need: 4 products total. Can we do it in fewer than 4?
    # Let U = [u00, u01; u10, u11] (2x2), V = [v0; v1] (2x1), W = [w0, w1] (1x2)
    # Then W[0] = u00*v0 + u01*v1, W[1] = u10*v0 + u11*v1
    # We need: W[0] = A[0,j1]*B[j1,k] + A[0,j2]*B[j2,k]
    #          W[1] = A[1,j1]*B[j1,k] + A[1,j2]*B[j2,k]
    # So u00=A[0,j1], u01=A[0,j2], u10=A[1,j1], u11=A[1,j2], v0=B[j1,k], v1=B[j2,k]
    # That's rank 2? No, it's a single rank-1 term for each output, so rank 2 total.
    # Wait, no. A rank-1 term is u⊗v⊗w. Here we have two separate rank-1 terms:
    # Term 1: u=[A[0,j1], A[1,j1], 0, 0] (for i=0,j1 and i=1,j1), v=[B[j1,k]], w=[1,0] (for k,0 and k,1)
    # Term 2: u=[A[0,j2], A[1,j2], 0, 0], v=[B[j2,k]], w=[1,0]
    # Hmm, this doesn't work cleanly. Let me think again.
    
    # Actually for 2x2x1, the standard decomposition is rank 4 (schoolbook).
    # But can we do better? The tensor has 4 non-zero entries (2 outputs x 2 inputs).
    # The rank of a 2x2x1 tensor... it's essentially two independent 2x1 multiplications.
    # I believe rank 4 is minimal for 2x2x1.
    
    # So the question is how to partition the 5x6 B matrix into blocks to minimize total rank.
    
    # Let's try: partition j into {0,1}, {2,3}, {4} and k into {0,1}, {2,3}, {4,5}
    # Block (j-block, k-block) has size (2 x |j-block| x |k-block|)
    # Rank for 2xa x b:
    #   2x2x2: 7 (Strassen)
    #   2x2x1: 4
    #   2x1x2: 4
    #   2x1x1: 2
    #   2x3x2: ? Let's compute: partition j into {j1,j2,j3}, k into {k1,k2}
    #     = 2x2x2 (rank 7) + 2x1x2 (rank 4) = 11, or 2x1x2 + 2x1x2 + 2x1x2 = 12
    #     So 11 is better.
    #   2x3x3: = 2x2x2 (7) + 2x1x2 (4) + 2x2x1 (4) + 2x1x1 (2) = 17, or other splits
    #     = 2x2x3 + 2x1x3 = (7+4) + (4+4) = 19? No, 2x2x3 = 2x2x2+2x2x1 = 7+4=11, 2x1x3=2x1x2+2x1x1=4+2=6, total 17
    #     Or 2x3x3 = 2x3x2 + 2x3x1 = 11 + (4+4+4)=12 = 23? No, 2x3x1 = 2x2x1+2x1x1 = 4+2=6, so 11+6=17
    #     Or 2x3x3 = 3 * 2x1x3 = 3*6=18, or 2x2x3 + 2x1x3 = 11+6=17
    #     Best is 17.
    
    # Let's enumerate partitions of {0,1,2,3,4} into contiguous blocks and {0,1,2,3,4,5} into contiguous blocks.
    
    def partition_options(n_elems):
        """Generate all ways to partition {0,...,n_elems-1} into contiguous blocks."""
        if n_elems == 1:
            return [[(0, 0)]]
        results = []
        for first_block_end in range(n_elems):
            rest = partition_options(n_elems - first_block_end - 1)
            for r in rest:
                # Shift r by first_block_end + 1
                shifted_r = [(a + first_block_end + 1, b + first_block_end + 1) for a, b in r]
                results.append([(0, first_block_end)] + shifted_r)
        return results
    
    def block_rank(n_rows, n_cols):
        """Return the minimal rank for 2 x n_rows x n_cols tensor using known block decompositions."""
        if n_rows == 1 and n_cols == 1:
            return 2
        if n_rows == 1 and n_cols == 2:
            return 4
        if n_rows == 2 and n_cols == 1:
            return 4
        if n_rows == 2 and n_cols == 2:
            return 7
        if n_rows == 3 and n_cols == 1:
            return 6
        if n_rows == 1 and n_cols == 3:
            return 6
        if n_rows == 2 and n_cols == 3:
            return 11
        if n_rows == 3 and n_cols == 2:
            return 11
        if n_rows == 3 and n_cols == 3:
            return 17
        if n_rows == 4 and n_cols == 1:
            return 8
        if n_rows == 1 and n_cols == 4:
            return 8
        if n_rows == 4 and n_cols == 2:
            return 15
        if n_rows == 2 and n_cols == 4:
            return 15
        if n_rows == 4 and n_cols == 3:
            return 22
        if n_rows == 3 and n_cols == 4:
            return 22
        if n_rows == 5 and n_cols == 1:
            return 10
        if n_rows == 1 and n_cols == 5:
            return 10
        if n_rows == 5 and n_cols == 2:
            return 18
        if n_rows == 2 and n_cols == 5:
            return 18
        if n_rows == 5 and n_cols == 3:
            return 28
        if n_rows == 3 and n_cols == 5:
            return 28
        if n_rows == 5 and n_cols == 4:
            return 36
        if n_rows == 4 and n_cols == 5:
            return 36
        if n_rows == 5 and n_cols == 5:
            return 46
        if n_rows == 5 and n_cols == 6:
            return 54
        # Fallback: schoolbook
        return 2 * n_rows * n_cols
    
    # Now find the best partition
    j_partitions = partition_options(m)
    k_partitions = partition_options(p)
    
    best_rank = float('inf')
    best_j_part = None
    best_k_part = None
    
    for jp in j_partitions:
        for kp in k_partitions:
            total_rank = 0
            for (j1, j2) in jp:
                for (k1, k2) in kp:
                    h = j2 - j1 + 1
                    w = k2 - k1 + 1
                    total_rank += block_rank(h, w)
            if total_rank < best_rank:
                best_rank = total_rank
                best_j_part = jp
                best_k_part = kp
    
    # Now build the decomposition using the best partition
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_step(u_pairs, v_pairs, w_pairs):
        nonlocal col
        if col >= best_rank:
            return False
        if time.time() - start_time > deadline:
            return False
        for idx, val in u_pairs:
            U[idx][col] = val
        for idx, val in v_pairs:
            V[idx][col] = val
        for idx, val in w_pairs:
            W[idx][col] = val
        col += 1
        return True
    
    def add_strassen(j1, j2, k1, k2):
        add_rank_step([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                     [(v_idx(j1,k1), 1)], 
                     [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
        add_rank_step([(u_idx(0,j1), 1)], 
                     [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                     [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
        add_rank_step([(u_idx(1,j2), 1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                     [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
        add_rank_step([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                     [(v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
        add_rank_step([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                     [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                     [(w_idx(k2,1), 1)])
        add_rank_step([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                     [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                     [(w_idx(k1,0), 1)])
    
    def add_schoolbook_2xa_x_b(j1, j2, k1, k2):
        """Add schoolbook terms for 2 x (j1..j2) x (k1..k2)."""
        for j in range(j1, j2+1):
            for k in range(k1, k2+1):
                for i in range(n):
                    add_rank_step([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])
    
    # Build the decomposition
    for (j1, j2) in best_j_part:
        for (k1, k2) in best_k_part:
            h = j2 - j1 + 1
            w = k2 - k1 + 1
            if h == 2 and w == 2:
                add_strassen(j1, j2, k1, k2)
            else:
                add_schoolbook_2xa_x_b(j1, j2, k1, k2)
    
    # Verify
    if time.time() - start_time < deadline - 5:
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check: for each (i,j,k), sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(best_rank):
                        s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        # If error is too large, fall back to known-good decomposition
        if max_err > 1e-10:
            # Fall back to the original 54-rank decomposition
            rank = 54
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            def add_rank_step_fb(u_pairs, v_pairs, w_pairs):
                nonlocal col
                for idx, val in u_pairs:
                    U[idx][col] = val
                for idx, val in v_pairs:
                    V[idx][col] = val
                for idx, val in w_pairs:
                    W[idx][col] = val
                col += 1
            def add_strassen_fb(j1, j2, k1, k2):
                add_rank_step_fb([(u_idx(0,j1), 1), (u_idx(1,j2), 1)], 
                                [(v_idx(j1,k1), 1), (v_idx(j2,k2), 1)], 
                                [(w_idx(k1,0), 1), (w_idx(k2,1), 1)])
                add_rank_step_fb([(u_idx(1,j1), 1), (u_idx(1,j2), 1)], 
                                [(v_idx(j1,k1), 1)], 
                                [(w_idx(k1,1), 1), (w_idx(k2,1), -1)])
                add_rank_step_fb([(u_idx(0,j1), 1)], 
                                [(v_idx(j1,k2), 1), (v_idx(j2,k2), -1)], 
                                [(w_idx(k2,0), 1), (w_idx(k2,1), 1)])
                add_rank_step_fb([(u_idx(1,j2), 1)], 
                                [(v_idx(j2,k1), 1), (v_idx(j1,k1), -1)], 
                                [(w_idx(k1,0), 1), (w_idx(k1,1), 1)])
                add_rank_step_fb([(u_idx(0,j1), 1), (u_idx(0,j2), 1)], 
                                [(v_idx(j2,k2), 1)], 
                                [(w_idx(k1,0), -1), (w_idx(k2,0), 1)])
                add_rank_step_fb([(u_idx(1,j1), 1), (u_idx(0,j1), -1)], 
                                [(v_idx(j1,k1), 1), (v_idx(j1,k2), 1)], 
                                [(w_idx(k2,1), 1)])
                add_rank_step_fb([(u_idx(0,j2), 1), (u_idx(1,j2), -1)], 
                                [(v_idx(j2,k1), 1), (v_idx(j2,k2), 1)], 
                                [(w_idx(k1,0), 1)])
            def add_schoolbook_fb(j, k1, k2):
                for k in range(k1, k2+1):
                    for i in range(n):
                        add_rank_step_fb([(u_idx(i,j), 1)], [(v_idx(j,k), 1)], [(w_idx(k,i), 1)])
            add_strassen_fb(0, 1, 0, 1)
            add_strassen_fb(0, 1, 2, 3)
            add_strassen_fb(0, 1, 4, 5)
            add_strassen_fb(2, 3, 0, 1)
            add_schoolbook_fb(4, 0, 1)
            add_strassen_fb(2, 3, 2, 3)
            add_schoolbook_fb(4, 2, 3)
            add_strassen_fb(2, 3, 4, 5)
            add_schoolbook_fb(4, 4, 5)
    
    return {"U": U, "V": V, "W": W}
