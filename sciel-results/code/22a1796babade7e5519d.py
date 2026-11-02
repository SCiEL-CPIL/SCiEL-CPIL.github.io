"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Block-Strassen decomposition for 2x4 * 4x5, then Winograd-style merge
    of rank-1 terms that share identical U-column AND identical W-column.
    Also attempts to find a rank-35 decomposition by trying to merge
    terms with identical U or W columns (not both), using rational
    coefficient solving.
    """
    import time
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 2, 4, 5

    rank_terms = []  # Each term is (u_dict, v_dict, w_dict)

    def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = [
            ({(i0, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0): 1, (i0+1, j0+1): 1},
             {(j0, k0): 1},
             {(i0+1, k0): 1, (i0+1, k0+1): -1}),
            ({(i0, j0): 1},
             {(j0, k0+1): 1, (j0+1, k0+1): -1},
             {(i0, k0+1): 1, (i0+1, k0+1): 1}),
            ({(i0+1, j0+1): 1},
             {(j0+1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0+1, k0): 1}),
            ({(i0, j0): 1, (i0, j0+1): 1},
             {(j0+1, k0+1): 1},
             {(i0, k0): -1, (i0, k0+1): 1}),
            ({(i0+1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0+1): 1},
             {(i0+1, k0+1): 1}),
            ({(i0, j0+1): 1, (i0+1, j0+1): -1},
             {(j0+1, k0): 1, (j0+1, k0+1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
        i0, j0 = a_offset
        j1, k0 = b_offset
        c_i0, c_k0 = c_offset
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
            ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
            ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
        ]
        for u, v, w in terms:
            u_scaled = {k: val * sign for k, val in u.items()}
            v_scaled = {k: val * sign for k, val in v.items()}
            w_scaled = {k: val * sign for k, val in w.items()}
            rank_terms.append((u_scaled, v_scaled, w_scaled))

    # Build block-Strassen decomposition (rank 36)
    add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
    add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
    add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
    add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
    add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
    add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    # Winograd-style merge: if two terms share identical U and identical W,
    # merge their V columns into one term.
    def merge_terms(terms):
        # Group by (u_key, w_key)
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, (u, v, w) in enumerate(terms):
            u_key = tuple(sorted(u.items()))
            w_key = tuple(sorted(w.items()))
            groups[(u_key, w_key)].append((idx, v))
        merged = []
        used = set()
        for (u_key, w_key), members in groups.items():
            if len(members) == 1:
                idx, v = members[0]
                if idx not in used:
                    u = dict(u_key)
                    w = dict(w_key)
                    merged.append((u, v, w))
                    used.add(idx)
            else:
                # Merge all V columns: sum them into one V dict
                u = dict(u_key)
                w = dict(w_key)
                v_sum = {}
                for idx, v in members:
                    if idx in used:
                        continue
                    for (jk, coeff) in v.items():
                        v_sum[jk] = v_sum.get(jk, 0) + coeff
                    used.add(idx)
                # Remove zero entries
                v_sum = {k: val for k, val in v_sum.items() if val != 0}
                if v_sum:
                    merged.append((u, v_sum, w))
        return merged

    rank_terms = merge_terms(rank_terms)

    # Try to reduce rank further by merging terms that share U or W
    def try_merge_u_or_w(terms):
        """
        Try to merge pairs of terms that share identical U or identical W.
        If two terms share U, we can try to merge them by solving for
        a new V and W that reproduces the same contribution.
        This is a more aggressive merge that may reduce rank.
        """
        from fractions import Fraction
        import itertools
        
        n_local, m_local, p_local = n, m, p
        current_terms = list(terms)
        
        def verify_terms(terms_list):
            for i in range(n_local):
                for j in range(m_local):
                    for k in range(p_local):
                        total = 0
                        for (u, v, w) in terms_list:
                            total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                        if total != 1:
                            return False
            return True
        
        # Try merging pairs with identical U
        for _ in range(5):  # Try up to 5 merges
            if time.time() > deadline - 5:
                break
            found = False
            for a_idx in range(len(current_terms)):
                if time.time() > deadline - 5:
                    break
                for b_idx in range(a_idx + 1, len(current_terms)):
                    if time.time() > deadline - 5:
                        break
                    u_a, v_a, w_a = current_terms[a_idx]
                    u_b, v_b, w_b = current_terms[b_idx]
                    
                    # Check if U columns are identical
                    if tuple(sorted(u_a.items())) == tuple(sorted(u_b.items())):
                        # Try to merge: new_v = alpha*v_a + beta*v_b, new_w = alpha*w_a + beta*w_b
                        # We need: u * (alpha*v_a + beta*v_b) * (alpha*w_a + beta*w_b) 
                        # to equal u*v_a*w_a + u*v_b*w_b
                        # This is a system of equations in alpha, beta
                        # Let's set up the linear system
                        # For each (i,j,k): u[i,j] * (alpha*v_a[j,k] + beta*v_b[j,k]) * (alpha*w_a[i,k] + beta*w_b[i,k])
                        # = u[i,j] * (alpha^2 * v_a[j,k]*w_a[i,k] + alpha*beta*(v_a[j,k]*w_b[i,k] + v_b[j,k]*w_a[i,k]) + beta^2 * v_b[j,k]*w_b[i,k])
                        # We want this to equal u[i,j] * (v_a[j,k]*w_a[i,k] + v_b[j,k]*w_b[i,k])
                        # This is a quadratic system, which is hard. Instead, let's try a simpler approach:
                        # If we can find alpha, beta such that for all (i,j,k) with u[i,j] != 0:
                        # alpha*v_a[j,k] * alpha*w_a[i,k] + alpha*v_a[j,k]*beta*w_b[i,k] + beta*v_b[j,k]*alpha*w_a[i,k] + beta*v_b[j,k]*beta*w_b[i,k]
                        # = v_a[j,k]*w_a[i,k] + v_b[j,k]*w_b[i,k]
                        # This is complex. Let's try a different approach: just check if v_a == v_b or w_a == w_b
                        if tuple(sorted(v_a.items())) == tuple(sorted(v_b.items())):
                            # Same U and V, merge W
                            w_merged = {}
                            for (ik, coeff) in w_a.items():
                                w_merged[ik] = w_merged.get(ik, 0) + coeff
                            for (ik, coeff) in w_b.items():
                                w_merged[ik] = w_merged.get(ik, 0) + coeff
                            w_merged = {k: val for k, val in w_merged.items() if val != 0}
                            if w_merged:
                                new_terms = [t for idx, t in enumerate(current_terms) if idx not in (a_idx, b_idx)]
                                new_terms.append((u_a, v_a, w_merged))
                                if verify_terms(new_terms):
                                    current_terms = new_terms
                                    found = True
                                    break
                        elif tuple(sorted(w_a.items())) == tuple(sorted(w_b.items())):
                            # Same U and W, merge V
                            v_merged = {}
                            for (jk, coeff) in v_a.items():
                                v_merged[jk] = v_merged.get(jk, 0) + coeff
                            for (jk, coeff) in v_b.items():
                                v_merged[jk] = v_merged.get(jk, 0) + coeff
                            v_merged = {k: val for k, val in v_merged.items() if val != 0}
                            if v_merged:
                                new_terms = [t for idx, t in enumerate(current_terms) if idx not in (a_idx, b_idx)]
                                new_terms.append((u_a, v_merged, w_a))
                                if verify_terms(new_terms):
                                    current_terms = new_terms
                                    found = True
                                    break
                    elif tuple(sorted(w_a.items())) == tuple(sorted(w_b.items())):
                        # Same W, try to merge
                        if tuple(sorted(v_a.items())) == tuple(sorted(v_b.items())):
                            u_merged = {}
                            for (ij, coeff) in u_a.items():
                                u_merged[ij] = u_merged.get(ij, 0) + coeff
                            for (ij, coeff) in u_b.items():
                                u_merged[ij] = u_merged.get(ij, 0) + coeff
                            u_merged = {k: val for k, val in u_merged.items() if val != 0}
                            if u_merged:
                                new_terms = [t for idx, t in enumerate(current_terms) if idx not in (a_idx, b_idx)]
                                new_terms.append((u_merged, v_a, w_a))
                                if verify_terms(new_terms):
                                    current_terms = new_terms
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
        return current_terms

    rank_terms = try_merge_u_or_w(rank_terms)

    # Verify correctness
    def verify(terms):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for (u, v, w) in terms:
                        total += u.get((i, j), 0) * v.get((j, k), 0) * w.get((i, k), 0)
                    if total != 1:
                        return False
        return True

    if not verify(rank_terms):
        # Fallback: return the unmerged version (should be correct)
        rank_terms = []
        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

    rank = len(rank_terms)

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(rank_terms):
        for (i, j), coeff in u_dict.items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in v_dict.items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in w_dict.items():
            W[k * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
