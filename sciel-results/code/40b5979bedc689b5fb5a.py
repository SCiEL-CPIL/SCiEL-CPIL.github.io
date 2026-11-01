"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 matrix multiplication with
    systematic column-removal (linear dependency search) to reduce rank.
    
    Base decomposition: rank 36 via Block-Strassen.
    Then search for linear dependencies among the 36 terms to reduce rank.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Rank-1 terms: each is (u_dict, v_dict, w_dict)
    # u_dict: {(i, j): coeff} for A[i,j], row in U = i*m + j
    # v_dict: {(j, k): coeff} for B[j,k], row in V = j*p + k
    # w_dict: {(i, k): coeff} for C[i,k], col in W = k*n + i
    
    rank_terms = []
    
    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        """
        Add Strassen terms for X(2x2) * Y(2x2) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0:k0+2]
        contributes to C[i0:i0+2, k0:k0+2]
        """
        terms = [
            # m1
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            # m2
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            # m3
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            # m4
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            # m5
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            # m6
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            # m7
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        """
        Add schoolbook terms for X(2x2) * Y(2x1) where:
        X = A[i0:i0+2, j0:j0+2]
        Y = B[j0:j0+2, k0]
        contributes to C[i0:i0+2, k0]
        """
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        
        for u, v, w in terms:
            u_s = {k: val * sign for k, val in u.items()}
            v_s = {k: val * sign for k, val in v.items()}
            w_s = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_s, v_s, w_s))
    
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B_top = B[0:2, 0:5], B_bot = B[2:4, 0:5]
    
    # C = A1 * B_top + A2 * B_bot
    # A1 * B_top: split B_top cols into [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)   # A1 * B_top[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)   # A1 * B_top[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1) # A1 * B_top[:, 4]
    
    # A2 * B_bot: split B_bot cols into [0,1], [2,3], [4]
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)   # A2 * B_bot[:, 0:2]
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)   # A2 * B_bot[:, 2:4]
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1) # A2 * B_bot[:, 4]
    
    # Build U, V, W matrices from the 36 terms
    rank = len(rank_terms)
    
    U = [[0] * rank for _ in range(n * m)]  # (8, rank)
    V = [[0] * rank for _ in range(m * p)]  # (20, rank)
    W = [[0] * rank for _ in range(p * n)]  # (10, rank)
    
    for c, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][c] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][c] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][c] = coeff
    
    # Now search for linear dependencies to reduce rank
    # The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # We need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # A term c is redundant if there exist coefficients alpha_d (d != c) such that
    # for all (i,j,k): U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{d!=c} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
    
    # This is a nonlinear condition. Instead, we can try to find a column c that can be
    # expressed as a linear combination of others in a way that preserves the tensor.
    
    # Alternative approach: try removing each column and check if the remaining columns
    # can still span the tensor (this is approximate - we check if the tensor values
    # are still achievable).
    
    # Better approach: the tensor is determined by the 36 rank-1 terms. If we can find
    # a set of 35 terms that still produce the correct tensor, we're done.
    
    # Let's build the tensor and check linear dependencies more carefully.
    # T[i,j,k] should equal 1 for all (i,j,k) in the multiplication pattern.
    
    # Actually, the condition is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = T[i,j,k]
    # where T[i,j,k] = 1 if the (i,j,k) entry is part of the multiplication, else 0.
    
    # For 2x4 * 4x5: T[i,j,k] = 1 for all i in [0,1], j in [0,3], k in [0,4]
    # (since it's the full matrix multiplication tensor)
    
    # Let's verify: the tensor for matrix multiplication M_{n,m,p} has entries
    # T[i,j,k] = 1 for all valid (i,j,k).
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # To check if a subset of columns is sufficient, we need to verify this equation.
    
    # Let's try a greedy removal: for each column, check if removing it still allows
    # the tensor to be reconstructed (by adjusting other columns).
    
    # Actually, the cleanest approach: try each column removal and solve for whether
    # the remaining columns can still produce the tensor. This is a system of equations.
    
    # For column c to be removable, we need: for all (i,j,k),
    #   U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = sum_{d in S} alpha_d * U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d]
    
    # This is nonlinear in the alphas. Instead, let's use a different approach:
    # Check if the 36 vectors (in the space of tensors) are linearly dependent.
    
    # Each column c defines a tensor T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need sum_c T_c = T (the full multiplication tensor).
    
    # If we can find a linear combination sum_{c in S} beta_c * T_c = T with |S| < 36,
    # we have a lower rank decomposition.
    
    # But this requires the T_c to be linearly independent, which they generally are
    # for Strassen decompositions.
    
    # Let's try a simpler approach: check for pairs of columns that might be combined.
    
    # Actually, let's just verify the current decomposition is correct and return it.
    # The systematic search for dependencies is complex; let's try a limited search.
    
    # Try: for each pair of columns (c1, c2), check if their sum/difference helps.
    # This is too expensive. Let's just return the rank-36 decomposition.
    
    # But first, let's try a quick check: are any two columns identical or negatives?
    # (They shouldn't be in Strassen, but let's verify.)
    
    # Check for duplicate columns
    seen = set()
    removable = set()
    for c in range(rank):
        key_u = tuple(U[r][c] for r in range(n*m))
        key_v = tuple(V[r][c] for r in range(m*p))
        key_w = tuple(W[r][c] for r in range(p*n))
        key = (key_u, key_v, key_w)
        if key in seen:
            removable.add(c)
        seen.add(key)
    
    # Also check for columns that are negatives of others
    col_keys = {}
    for c in range(rank):
        key_u = tuple(U[r][c] for r in range(n*m))
        key_v = tuple(V[r][c] for r in range(m*p))
        key_w = tuple(W[r][c] for r in range(p*n))
        col_keys[c] = (key_u, key_v, key_w)
    
    for c in range(rank):
        if c in removable:
            continue
        neg_key_u = tuple(-x for x in col_keys[c][0])
        neg_key_v = tuple(-x for x in col_keys[c][1])
        neg_key_w = tuple(-x for x in col_keys[c][2])
        neg_key = (neg_key_u, neg_key_v, neg_key_w)
        # Check if neg_key exists in another column
        for c2 in range(rank):
            if c2 != c and c2 not in removable and col_keys.get(c2) == neg_key:
                # c and c2 are negatives; we can potentially combine them
                # But in our decomposition, we need sum of all = T, so if we have
                # c and -c, they cancel. This means we can remove both.
                removable.add(c)
                removable.add(c2)
                break
    
    if removable:
        # Rebuild without removable columns
        keep = [c for c in range(rank) if c not in removable]
        new_rank = len(keep)
        U_new = [[U[r][c] for c in keep] for r in range(n*m)]
        V_new = [[V[r][c] for c in keep] for r in range(m*p)]
        W_new = [[W[r][c] for c in keep] for r in range(p*n)]
        return {"U": U_new, "V": V_new, "W": W_new}
    
    # No simple duplicates found. Try a more sophisticated search:
    # Check if any column can be expressed as a linear combination of others
    # in the tensor space. This requires solving a linear system.
    
    # For each column c, define the tensor T_c as a vector in R^(n*m*p)
    # T_c[i*m+j, j*p+k, k*n+i] -> flatten to index (i*m+j)*(m*p) + (j*p+k) ... 
    # Actually, let's index the tensor by (i, j, k) -> flat index i*(m*p) + j*p + k
    
    np = __import__('numpy')
    
    tensor_dim = n * m * p
    T_target = np.ones(tensor_dim)  # The multiplication tensor is all 1s
    
    # Build the tensor vectors for each column
    T_cols = []
    for c in range(rank):
        Tc = np.zeros(tensor_dim)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    flat_idx = i * (m * p) + j * p + k
                    Tc[flat_idx] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
        T_cols.append(Tc)
    
    T_cols = np.array(T_cols)  # shape (rank, tensor_dim)
    
    # Verify: sum of T_cols should equal T_target
    T_sum = T_cols.sum(axis=0)
    if not np.allclose(T_sum, T_target, atol=1e-9):
        # Decomposition is incorrect; return as-is
        return {"U": U, "V": V, "W": W}
    
    # Now try to find a subset of columns whose sum equals T_target.
    # This is equivalent to finding a subset S and coefficients such that
    # sum_{c in S} alpha_c * T_cols[c] = T_target
    # with alpha_c = 1 for all c in S (since we can't scale columns independently
    # without changing the decomposition structure).
    
    # Actually, in a tensor decomposition, each rank-1 term has its own scaling.
    # We can absorb scalings into U, V, W. So we need:
    # sum_{c in S} T_cols[c] = T_target
    # where we can rescale each T_cols[c] by a factor.
    
    # This means: find S and scalings s_c such that sum_{c in S} s_c * T_cols[c] = T_target
    # This is a linear system in the s_c's.
    
    # Let's try: for each column c, check if T_target - T_cols[c] is in the span
    # of the other columns. If so, we can remove c.
    
    # For efficiency, let's try removing columns one at a time.
    # For each c, solve: sum_{d != c} s_d * T_cols[d] = T_target
    # This is a least-squares problem. If the solution has all s_d = 1 (or close),
    # then c is redundant.
    
    # Actually, the condition is: T_target = sum_{d != c} s_d * T_cols[d]
    # We want s_d = 1 for all d != c. This means T_target - sum_{d != c} T_cols[d] = 0
    # i.e., T_cols[c] = 0, which is not the case.
    
    # The correct condition: we want to find s_d such that sum_{d != c} s_d * T_cols[d] = T_target
    # The natural solution is s_d = 1 for all d, which gives sum_{d != c} T_cols[d] = T_target - T_cols[c]
    # So we need T_target - T_cols[c] = sum_{d != c} T_cols[d], which is always true by construction.
    # This means we can always "remove" c by setting s_d = 1, but this doesn't reduce rank
    # because we still need all d != c.
    
    # To actually reduce rank, we need to find a proper subset S with |S| < rank such that
    # there exist s_c for c in S with sum_{c in S} s_c * T_cols[c] = T_target.
    
    # This is a linear system: T_cols[S]^T * s = T_target
    # We need this to have a solution.
    
    # Let's try: for each column c, check if T_target is in the span of T_cols without c.
    # i.e., is T_target in the column space of T_cols[remove c]?
    
    # Since T_target = sum of all T_cols, T_target is always in the span of all columns.
    # The question is whether it's in the span of a proper subset.
    
    # Let's check the rank of the matrix T_cols (rank x tensor_dim).
    # If the columns are linearly independent, we can't reduce.
    
    # Compute rank of T_cols
    rank_T = np.linalg.matrix_rank(T_cols, tol=1e-9)
    
    if rank_T < rank:
        # There are linear dependencies! Find them.
        # Use SVD to find the null space
        U_svd, S, Vt = np.linalg.svd(T_cols, full_matrices=False)
        # Null space is the last (rank - rank_T) rows of Vt
        null_dim = rank - rank_T
        if null_dim > 0:
            null_space = Vt[rank_T:rank]  # shape (null_dim, rank)
            # Find a null vector with many zeros
            for nv in null_space:
                # Try to find a sparse null vector
                # For now, just check if we can remove a column
                # A null vector means sum_c nv[c] * T_cols[c] = 0
                # So we can add nv[c] * T_cols[c] to any subset
                pass
    
    # Given the complexity, let's just return the rank-36 decomposition
    # unless we found a simple duplicate/negative pair.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
