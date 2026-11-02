"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.
    Uses block-Strassen (rank 36) then merges terms with identical U-factors
    or V-factors to reduce rank.
    """
    import time
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit

    n, m, p = DIMENSIONS  # 2, 4, 5

    def build_base_terms():
        """Build the base 36-term block-Strassen decomposition."""
        terms = []

        def add_strassen_2x2x2(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset

            t = [
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
            for u, v, w in t:
                u_s = {k: val * sign for k, val in u.items()}
                v_s = {k: val * sign for k, val in v.items()}
                w_s = {k: val * sign for k, val in w.items()}
                terms.append((u_s, v_s, w_s))

        def add_schoolbook_2x2x1(a_offset, b_offset, c_offset, sign=1):
            i0, j0 = a_offset
            j1, k0 = b_offset
            c_i0, c_k0 = c_offset

            t = [
                ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
                ({(i0, j0+1): 1}, {(j0+1, k0): 1}, {(i0, k0): 1}),
                ({(i0+1, j0): 1}, {(j0, k0): 1}, {(i0+1, k0): 1}),
                ({(i0+1, j0+1): 1}, {(j0+1, k0): 1}, {(i0+1, k0): 1}),
            ]
            for u, v, w in t:
                u_s = {k: val * sign for k, val in u.items()}
                v_s = {k: val * sign for k, val in v.items()}
                w_s = {k: val * sign for k, val in w.items()}
                terms.append((u_s, v_s, w_s))

        add_strassen_2x2x2((0, 0), (0, 0), (0, 0))
        add_strassen_2x2x2((0, 2), (2, 0), (0, 0))
        add_strassen_2x2x2((0, 0), (0, 2), (0, 2))
        add_schoolbook_2x2x1((0, 0), (0, 4), (0, 4))
        add_strassen_2x2x2((0, 2), (2, 2), (0, 2))
        add_schoolbook_2x2x1((0, 2), (2, 4), (0, 4))

        return terms

    def dict_key(d):
        return tuple(sorted(d.items()))

    def merge_by_factor(terms, factor_idx):
        """Merge terms that have identical factor at factor_idx (0=U, 1=V)."""
        groups = {}
        for idx, term in enumerate(terms):
            key = dict_key(term[factor_idx])
            if key not in groups:
                groups[key] = []
            groups[key].append(idx)

        merged = []
        for key, indices in groups.items():
            if len(indices) > 1:
                # Combine the other two factors
                ref_term = terms[indices[0]]
                other_factors = [f for f in range(3) if f != factor_idx]
                combined = [dict() for _ in other_factors]
                for idx in indices:
                    for oi, f in enumerate(other_factors):
                        for k, val in terms[idx][f].items():
                            combined[oi][k] = combined[oi].get(k, 0) + val
                # Remove zeros
                cleaned = [dict((k, v) for k, v in c.items() if v != 0) for c in combined]
                new_term = [None, None, None]
                new_term[factor_idx] = ref_term[factor_idx]
                for oi, f in enumerate(other_factors):
                    new_term[f] = cleaned[oi]
                merged.append(tuple(new_term))
            else:
                merged.append(terms[indices[0]])
        return merged

    def verify(terms):
        """Verify the decomposition is correct."""
        import random
        random.seed(12345)
        A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
        AB = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]

        rank = len(terms)
        for i in range(n):
            for k in range(p):
                val = 0
                for col in range(rank):
                    u, v, w = terms[col]
                    u_val = sum(u.get((i, j), 0) * A[i][j] for j in range(m))
                    v_val = sum(v.get((j, k), 0) * B[j][k] for j in range(m))
                    w_val = w.get((i, k), 0)
                    val += u_val * v_val * w_val
                if val != AB[i][k]:
                    return False
        return True

    def build_matrices(terms):
        """Build U, V, W from terms."""
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for col, (u_dict, v_dict, w_dict) in enumerate(terms):
            for (i, j), coeff in u_dict.items():
                U[i * m + j][col] = coeff
            for (j, k), coeff in v_dict.items():
                V[j * p + k][col] = coeff
            for (i, k), coeff in w_dict.items():
                W[k * n + i][col] = coeff
        return U, V, W

    # Start with base terms
    terms = build_base_terms()
    best_terms = terms
    best_rank = len(terms)

    # Try merging by U factor
    merged_u = merge_by_factor(terms, 0)
    if len(merged_u) < best_rank and verify(merged_u):
        best_terms = merged_u
        best_rank = len(merged_u)

    # Try merging by V factor
    merged_v = merge_by_factor(terms, 1)
    if len(merged_v) < best_rank and verify(merged_v):
        best_terms = merged_v
        best_rank = len(merged_v)

    # Try sequential: merge by U then by V
    if len(merged_u) < len(terms):
        merged_uv = merge_by_factor(merged_u, 1)
        if len(merged_uv) < best_rank and verify(merged_uv):
            best_terms = merged_uv
            best_rank = len(merged_uv)

    # Try sequential: merge by V then by U
    if len(merged_v) < len(terms):
        merged_vu = merge_by_factor(merged_v, 0)
        if len(merged_vu) < best_rank and verify(merged_vu):
            best_terms = merged_vu
            best_rank = len(merged_vu)

    U, V, W = build_matrices(best_terms)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
