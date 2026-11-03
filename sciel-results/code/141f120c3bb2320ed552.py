import time
from collections import defaultdict

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 6

    strassen_7 = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)], [((0,0),1),((0,1),1),((1,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)], [((1,0),1),((1,1),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)], [((0,1),1),((1,1),1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)], [((0,0),1),((1,0),1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)], [((0,0),1),((1,0),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)], [((1,0),1),((1,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)], [((0,1),1),((1,1),1)]),
    ]

    terms = []

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        u = {i: c for i, c in u_coeffs if c != 0}
        v = {i: c for i, c in v_coeffs if c != 0}
        w = {i: c for i, c in w_coeffs if c != 0}
        if u and v and w:
            terms.append((u, v, w))

    def a_idx(block, r, c):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*4 + (y*2+c)

    def b_idx(block, r, c, col_offset=0):
        x, y = int(block[0]), int(block[1])
        return (x*2+r)*6 + (y*2+c+col_offset)

    def c_idx(block, r, c, col_offset=0):
        x, y = int(block[0]), int(block[1])
        return (y*2+c+col_offset)*4 + (x*2+r)

    strassen_2x2 = [
        ([((0,0),1),((1,1),1)], [((0,0),1),((1,1),1)]),
        ([((1,0),1),((1,1),1)], [((0,0),1)]),
        ([((0,0),1)], [((0,1),1),((1,1),-1)]),
        ([((1,1),1)], [((1,0),1),((0,0),-1)]),
        ([((0,0),1),((0,1),1)], [((1,1),1)]),
        ([((1,0),1),((0,0),-1)], [((0,0),1),((0,1),1)]),
        ([((0,1),1),((1,1),-1)], [((1,0),1),((1,1),1)]),
    ]

    result_map = {
        (0,0): [(0,1), (3,1), (4,-1), (6,1)],
        (0,1): [(2,1), (4,1)],
        (1,0): [(1,1), (3,1)],
        (1,1): [(0,1), (1,-1), (2,1), (5,1)],
    }

    big_combos = [
        ([('00',1),('11',1)], [('00',1),('11',1)]),
        ([('10',1),('11',1)], [('00',1)]),
        ([('00',1)], [('01',1),('11',-1)]),
        ([('11',1)], [('10',1),('00',-1)]),
        ([('00',1),('01',1)], [('11',1)]),
        ([('10',1),('00',-1)], [('00',1),('01',1)]),
        ([('01',1),('11',-1)], [('10',1),('11',1)]),
    ]

    big_result_map = {
        '00': [(0,1), (3,1), (4,-1), (6,1)],
        '01': [(2,1), (4,1)],
        '10': [(1,1), (3,1)],
        '11': [(0,1), (1,-1), (2,1), (5,1)],
    }

    # Part 1: Strassen for 4x4x4
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            x_coeffs, y_coeffs = strassen_2x2[s_idx]

            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))

            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b_idx(b_block, ry, cy, 0)
                    v_coeffs.append((idx, b_sign * cy_c))

            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    idx = c_idx(c_block, rc[0], rc[1], 0)
                    w_coeffs.append((idx, big_coeff * small_coeff))

            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Part 2: 4x4x2
    products = [
        ('00', 'T', 'top'),
        ('01', 'B', 'top'),
        ('10', 'T', 'bot'),
        ('11', 'B', 'bot'),
    ]

    for a_block, b_block, c_part in products:
        if time.time() > deadline:
            break
        for s_idx in range(7):
            if time.time() > deadline:
                break
            x_coeffs, y_coeffs = strassen_2x2[s_idx]

            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                x, y = int(a_block[0]), int(a_block[1])
                idx = (x*2+rx)*4 + (y*2+cx)
                u_coeffs.append((idx, cx_c))

            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                if b_block == 'T':
                    idx = ry*6 + (4+cy)
                else:
                    idx = (2+ry)*6 + (4+cy)
                v_coeffs.append((idx, cy_c))

            w_coeffs = []
            for rc, s_contribs in result_map.items():
                small_coeff = 0
                for s_check, s_c in s_contribs:
                    if s_check == s_idx:
                        small_coeff = s_c
                        break
                if small_coeff == 0:
                    continue
                if c_part == 'top':
                    idx = (4+rc[1])*4 + rc[0]
                else:
                    idx = (4+rc[1])*4 + (2+rc[0])
                w_coeffs.append((idx, small_coeff))

            add_term(u_coeffs, v_coeffs, w_coeffs)

    # Improved merging with more iterations
    def merge_by_uv(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(u.items())), tuple(sorted(v.items())))
            groups[key].append(w)
        merged = []
        for key, w_list in groups.items():
            w_sum = defaultdict(float)
            for w in w_list:
                for i, c in w.items():
                    w_sum[i] += c
            w_sum = {i: c for i, c in w_sum.items() if abs(c) > 1e-9}
            if w_sum:
                u = dict(key[0])
                v = dict(key[1])
                merged.append((u, v, w_sum))
        return merged

    def merge_by_uw(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(u.items())), tuple(sorted(w.items())))
            groups[key].append(v)
        merged = []
        for key, v_list in groups.items():
            v_sum = defaultdict(float)
            for v in v_list:
                for i, c in v.items():
                    v_sum[i] += c
            v_sum = {i: c for i, c in v_sum.items() if abs(c) > 1e-9}
            if v_sum:
                u = dict(key[0])
                w = dict(key[1])
                merged.append((u, v_sum, w))
        return merged

    def merge_by_vw(terms_in):
        groups = defaultdict(list)
        for u, v, w in terms_in:
            key = (tuple(sorted(v.items())), tuple(sorted(w.items())))
            groups[key].append(u)
        merged = []
        for key, u_list in groups.items():
            u_sum = defaultdict(float)
            for u in u_list:
                for i, c in u.items():
                    u_sum[i] += c
            u_sum = {i: c for i, c in u_sum.items() if abs(c) > 1e-9}
            if u_sum:
                v = dict(key[0])
                w = dict(key[1])
                merged.append((u_sum, v, w))
        return merged

    def try_rank_reduction(terms_in):
        """Try to find a term that is a linear combination of others."""
        import numpy as np
        rank = len(terms_in)
        if rank < 2:
            return terms_in

        # Build U, V, W matrices
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        for col, (u, v, w) in enumerate(terms_in):
            for i, c in u.items():
                U[i, col] = c
            for i, c in v.items():
                V[i, col] = c
            for i, c in w.items():
                W[i, col] = c

        # For each column j, check if it can be expressed as a linear combo of others
        # This is a non-trivial check. We use the fact that if column j of U is in span of
        # other columns, AND column j of V is in span of other columns, AND column j of W
        # is in span of other columns, then we might be able to eliminate it.
        # But actually, we need: there exist alpha_k such that for all i: U[i,j] = sum_k alpha_k * U[i,k]
        # AND for all i: V[i,j] = sum_k alpha_k * V[i,k]  (same alpha!)
        # AND for all i: W[i,j] = sum_k alpha_k * W[i,k]  (same alpha!)
        #
        # This is very restrictive. A simpler approach: just try pairwise merging more aggressively.

        # Alternative: try to find two terms that can be merged by completing the product
        # If term_a has u_a, v_a, w_a and term_b has u_b, v_b, w_b,
        # we can't merge them unless u_a == u_b or v_a == v_b or w_a == w_b.
        # The existing merge functions handle this.

        # Let's try a different approach: random perturbation and re-merge
        # This is a heuristic that might find better groupings.
        return terms_in

    current = terms
    for iteration in range(50):
        if time.time() > deadline:
            break
        merged_uv = merge_by_uv(current)
        merged_uw = merge_by_uw(current)
        merged_vw = merge_by_vw(current)
        best = min([merged_uv, merged_uw, merged_vw], key=len)
        if len(best) >= len(current):
            # Try all three and pick the best
            candidates = [merged_uv, merged_uw, merged_vw]
            best = min(candidates, key=len)
            if len(best) >= len(current):
                break
        current = best

    # Additional pass: try to find linear dependencies
    if time.time() < deadline - 5:
        try:
            import numpy as np
            rank = len(current)
            if rank > 1:
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))

                for col, (u, v, w) in enumerate(current):
                    for i, c in u.items():
                        U[i, col] = c
                    for i, c in v.items():
                        V[i, col] = c
                    for i, c in w.items():
                        W[i, col] = c

                # Check for exact linear dependencies in the Kronecker structure
                # For each pair (i,j), check if term j can be written as alpha * term i
                # in all three matrices simultaneously
                removed = [False] * rank
                for j in range(rank):
                    if removed[j]:
                        continue
                    if time.time() > deadline:
                        break
                    for i in range(rank):
                        if i == j or removed[i]:
                            continue
                        # Check if U[:,j] = alpha * U[:,i], V[:,j] = alpha * V[:,i], W[:,j] = alpha * W[:,i]
                        # Find alpha from U
                        mask = np.abs(U[:, i]) > 1e-9
                        if not np.any(mask):
                            continue
                        alpha_candidates = U[mask, j] / U[mask, i]
                        if len(alpha_candidates) == 0:
                            continue
                        alpha = alpha_candidates[0]
                        # Check consistency
                        if not np.allclose(U[mask, j], alpha * U[mask, i], atol=1e-9):
                            continue
                        # Check V
                        mask_v = np.abs(V[:, i]) > 1e-9
                        if np.any(mask_v):
                            if not np.allclose(V[mask_v, j], alpha * V[mask_v, i], atol=1e-9):
                                continue
                        # Check W
                        mask_w = np.abs(W[:, i]) > 1e-9
                        if np.any(mask_w):
                            if not np.allclose(W[mask_w, j], alpha * W[mask_w, i], atol=1e-9):
                                continue
                        # Term j is a scalar multiple of term i, remove j
                        removed[j] = True
                        break

                if any(removed):
                    current = [t for idx, t in enumerate(current) if not removed[idx]]
                    # Re-merge after removal
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        merged_uv = merge_by_uv(current)
                        merged_uw = merge_by_uw(current)
                        merged_vw = merge_by_vw(current)
                        best = min([merged_uv, merged_uw, merged_vw], key=len)
                        if len(best) >= len(current):
                            break
                        current = best
        except Exception:
            pass

    # Convert to matrices
    rank = len(current)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u, v, w) in enumerate(current):
        for i, c in u.items():
            U[i][col] = c
        for i, c in v.items():
            V[i][col] = c
        for i, c in w.items():
            W[i][col] = c

    return {"U": U, "V": V, "W": W}
