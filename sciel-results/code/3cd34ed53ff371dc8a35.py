"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5 matrix multiplication.
    Base rank 36. Then use a structured linear-algebraic search to find
    linear dependencies among the rank-1 tensor terms and remove redundant
    columns to reduce rank.
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

    # Build the base rank-36 decomposition
    # We use a construction that groups columns 2,3,4 into a 2x2x3 block
    # which has a known rank-10 decomposition (generated as 11 terms with 1 dependency).
    # A1 = A[0:2, 0:2], A2 = A[0:2, 2:4]
    # B_top = B[0:2, 0:5], B_bot = B[2:4, 0:5]
    # C = A1 * B_top + A2 * B_bot

    # A1 * B_top
    # Cols 0,1: Strassen 2x2x2
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    # Cols 2,3,4: 2x2x3 block (7 Strassen + 4 Schoolbook = 11 terms)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    # Note: The schoolbook for col 4 is part of the 2x2x3 block with cols 2,3.
    # The dependency search will handle the reduction.

    # A2 * B_bot
    # Cols 0,1: Strassen 2x2x2
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    # Cols 2,3,4: 2x2x3 block
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)

    rank = len(rank_terms)  # Should be 36

    # Build U, V, W matrices
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

    import numpy as np

    # Verify decomposition correctness
    tensor_dim = n * m * p
    T_target = np.ones(tensor_dim)

    T_cols = np.zeros((rank, tensor_dim))
    for c in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    flat_idx = i * (m * p) + j * p + k
                    T_cols[c, flat_idx] = U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]

    T_sum = T_cols.sum(axis=0)
    if not np.allclose(T_sum, T_target, atol=1e-9):
        # Decomposition is incorrect; return as-is
        return {"U": U, "V": V, "W": W}

    # Now search for linear dependencies to reduce rank.
    # Key insight: If we can find a non-trivial linear combination of the
    # rank-1 tensor terms that sums to zero, we can use it to eliminate one term.
    #
    # More practically: we look for a column c such that T_cols[c] is in the
    # span of the other columns. If so, we can remove c and adjust the others.
    #
    # But in a tensor decomposition, we can't arbitrarily rescale columns.
    # However, we CAN absorb scalings into U, V, W. So the question is:
    # Is there a subset S of columns with |S| < rank such that there exist
    # scalings s_c for c in S with sum_{c in S} s_c * T_cols[c] = T_target?
    #
    # This is a linear system: T_cols[S]^T @ s = T_target
    # We need this to have a solution.
    #
    # Strategy: Try removing columns one at a time (greedy). For each column c,
    # check if T_target is in the column space of T_cols without c.
    #
    # A more efficient approach: compute the rank of T_cols. If rank(T_cols) < rank,
    # there are linear dependencies. Find them and use them to reduce the set.

    # First, check for simple duplicates or negatives
    col_keys = {}
    for c in range(rank):
        key_u = tuple(U[r][c] for r in range(n*m))
        key_v = tuple(V[r][c] for r in range(m*p))
        key_w = tuple(W[r][c] for r in range(p*n))
        col_keys[c] = (key_u, key_v, key_w)

    removable = set()
    seen = set()
    for c in range(rank):
        key = col_keys[c]
        if key in seen:
            removable.add(c)
        seen.add(key)

    # Check for negative pairs
    for c in range(rank):
        if c in removable:
            continue
        neg_key_u = tuple(-x for x in col_keys[c][0])
        neg_key_v = tuple(-x for x in col_keys[c][1])
        neg_key_w = tuple(-x for x in col_keys[c][2])
        neg_key = (neg_key_u, neg_key_v, neg_key_w)
        for c2 in range(c+1, rank):
            if c2 not in removable and col_keys.get(c2) == neg_key:
                # c and c2 are negatives; they cancel in the sum
                removable.add(c)
                removable.add(c2)
                break

    if removable:
        keep = [c for c in range(rank) if c not in removable]
        new_rank = len(keep)
        U_new = [[U[r][c] for c in keep] for r in range(n*m)]
        V_new = [[V[r][c] for c in keep] for r in range(m*p)]
        W_new = [[W[r][c] for c in keep] for r in range(p*n)]
        return {"U": U_new, "V": V_new, "W": W_new}

    # Now try a more sophisticated search: greedy column removal with linear algebra
    # For each column c, check if T_target can be expressed as a linear combination
    # of the other columns.

    # Compute the rank of T_cols
    rank_T = np.linalg.matrix_rank(T_cols, tol=1e-9)

    if rank_T < rank:
        # There are linear dependencies. Find a null vector.
        U_svd, S, Vt = np.linalg.svd(T_cols, full_matrices=False)
        null_dim = rank - rank_T
        if null_dim > 0:
            null_space = Vt[rank_T:rank]  # shape (null_dim, rank)
            # Try to find a sparse null vector that allows removing a column
            for nv in null_space:
                # nv @ T_cols = 0, so sum_c nv[c] * T_cols[c] = 0
                # This means we can modify the decomposition by adding nv[c] * T_cols[c]
                # to any column. To remove column c, we need nv[c] != 0.
                # Then we can absorb the adjustment into the other columns.
                #
                # Specifically, if we remove column c, we need to find new coefficients
                # for the remaining columns. The null vector tells us how to do this.
                #
                # For now, just try removing the column with the largest |nv[c]|
                # and check if the remaining columns can still span T_target.

                # Find the column with largest |nv[c]|
                max_idx = np.argmax(np.abs(nv))
                if nv[max_idx] != 0:
                    # Try removing max_idx
                    keep = [c for c in range(rank) if c != max_idx]
                    T_keep = T_cols[keep]
                    # Check if T_target is in the column space of T_keep
                    # This is equivalent to checking if rank([T_keep | T_target]) == rank(T_keep)
                    augmented = np.column_stack([T_keep.T, T_target.reshape(-1, 1)])
                    rank_aug = np.linalg.matrix_rank(augmented, tol=1e-9)
                    rank_keep = np.linalg.matrix_rank(T_keep.T, tol=1e-9)
                    if rank_aug == rank_keep:
                        # T_target is in the span of T_keep
                        # Solve for the coefficients
                        # T_keep.T @ s = T_target
                        s, residuals, _, _ = np.linalg.lstsq(T_keep.T, T_target, rcond=None)
                        # Verify
                        T_recon = T_keep.T @ s
                        if np.allclose(T_recon, T_target, atol=1e-9):
                            # We found a valid decomposition with rank-1 terms
                            # Now we need to update U, V, W with the new scalings
                            U_new = [[U[r][c] for c in keep] for r in range(n*m)]
                            V_new = [[V[r][c] for c in keep] for r in range(m*p)]
                            W_new = [[W[r][c] for c in keep] for r in range(p*n)]
                            # Apply the scalings
                            for i, c in enumerate(keep):
                                scale = s[i]
                                # Scale only U to implement the linear combination
                                # T'_i = (scale * U_i) \otimes V_i \otimes W_i = scale * T_i
                                for r in range(n*m):
                                    U_new[r][i] *= scale
                            return {"U": U_new, "V": V_new, "W": W_new}

    # If no simple reduction found, try a greedy approach:
    # For each column, check if removing it still allows spanning T_target.
    for c in range(rank):
        if time.time() > deadline - 10:
            break
        keep = [d for d in range(rank) if d != c]
        T_keep = T_cols[keep]
        augmented = np.column_stack([T_keep.T, T_target.reshape(-1, 1)])
        rank_aug = np.linalg.matrix_rank(augmented, tol=1e-9)
        rank_keep = np.linalg.matrix_rank(T_keep.T, tol=1e-9)
        if rank_aug == rank_keep:
            # T_target is in the span of T_keep
            s, residuals, _, _ = np.linalg.lstsq(T_keep.T, T_target, rcond=None)
            T_recon = T_keep.T @ s
            if np.allclose(T_recon, T_target, atol=1e-9):
                U_new = [[U[r][d] for d in keep] for r in range(n*m)]
                V_new = [[V[r][d] for d in keep] for r in range(m*p)]
                W_new = [[W[r][d] for d in keep] for r in range(p*n)]
                for i, d in enumerate(keep):
                    scale = s[i]
                    # Scale only U to implement the linear combination
                    for r in range(n*m):
                        U_new[r][i] *= scale
                return {"U": U_new, "V": V_new, "W": W_new}

    # No reduction found; return the original rank-36 decomposition
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
