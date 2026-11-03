import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses randomized greedy + local swap improvement to find a maximum packing of
    disjoint 2x2x2 Strassen blocks, then attempts to merge pairs of terms.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_triples = n * m * p  # 105

    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k

    # Precompute all possible 2x2x2 blocks with bitmasks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask))

    # Search for maximum packing using randomized greedy + local swap
    best_num_blocks = 0
    best_packing = []

    num_trials = 2000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break

        # Randomized greedy
        shuffled_blocks = all_blocks[:]
        random.shuffle(shuffled_blocks)
        used_mask = 0
        packing = []

        for params, bm in shuffled_blocks:
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm

        # Local improvement: try to add a block by removing one
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            used_mask = 0
            for params in packing:
                bm = all_blocks[all_blocks.index((params, None))[0]][1] if False else None
                # Recompute used_mask
            used_mask = 0
            for params in packing:
                # Find the bitmask for this block
                for bp, bm in all_blocks:
                    if bp == params:
                        used_mask |= bm
                        break

            # Try adding each unused block
            for params, bm in all_blocks:
                if (bm & used_mask) == 0:
                    packing.append(params)
                    improved = True
                    break

            # Try swapping: remove one block, add two others
            if not improved and len(packing) > 0 and time.time() < deadline - 10.0:
                for remove_idx in range(len(packing)):
                    if time.time() > deadline - 10.0:
                        break
                    # Recompute mask without this block
                    mask_without = 0
                    for idx, params in enumerate(packing):
                        if idx == remove_idx:
                            continue
                        for bp, bm in all_blocks:
                            if bp == params:
                                mask_without |= bm
                                break

                    # Try to add two blocks that fit in the freed space
                    freed_mask = (1 << total_triples) - 1 - mask_without
                    addable = []
                    for params, bm in all_blocks:
                        if params not in packing and (bm & mask_without) == 0:
                            addable.append((params, bm))

                    # Try pairs from addable
                    for a in range(len(addable)):
                        for b in range(a + 1, len(addable)):
                            if (addable[a][1] & addable[b][1]) == 0:
                                packing.pop(remove_idx)
                                packing.append(addable[a][0])
                                packing.append(addable[b][0])
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break

        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]

            if best_num_blocks >= 13:  # Theoretical max
                break

    # Construct U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_entries, v_entries, w_entries):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2

        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2

        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2

        # M1 = (a11+a22)*(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        # M2 = (a21+a22)*b11
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        # M5 = (a11+a12)*b22
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        # M6 = (a21-a11)*(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        # M7 = (a12-a22)*(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

    # Merge pass: check if any pair of terms can be merged into one rank-1 term
    if time.time() < deadline - 15.0 and rank > 1:
        def can_merge(i, j):
            """Check if terms i and j can be replaced by a single rank-1 term."""
            u1, v1, w1 = U_cols[i], V_cols[i], W_cols[i]
            u2, v2, w2 = U_cols[j], V_cols[j], W_cols[j]

            # The sum tensor T[a,b,c] = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
            # For T to be rank-1, there must exist u,v,w such that T[a,b,c] = u[a]*v[b]*w[c]
            # This means T must be separable.
            # Check: for any a1,a2,b1,b2,c1,c2:
            # T[a1,b1,c1]*T[a2,b2,c2]*T[a1,b2,c1]*T[a2,b1,c2] == T[a1,b2,c2]*T[a2,b1,c1]*T[a1,b1,c2]*T[a2,b2,c1]
            # Simpler: check that the matricization has rank 1.
            # Matricize along first mode: M[a, (b,c)] = T[a,b,c]
            # Rank 1 iff all rows are proportional.

            # Collect non-zero (b,c) pairs where T has support
            bc_pairs = set()
            for a in range(U_rows):
                for b in range(V_rows):
                    for c in range(W_rows):
                        val = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
                        if val != 0:
                            bc_pairs.add((b, c))

            if not bc_pairs:
                return True, ([0]*U_rows, [0]*V_rows, [0]*W_rows)

            # Pick reference (b0,c0)
            b0, c0 = list(bc_pairs)[0]

            # For rank 1: T[a,b0,c0] = u[a]*v[b0]*w[c0]
            # So u[a] is proportional to T[a,b0,c0] for all a
            # Check consistency: T[a,b,c] / T[a,b0,c0] should be independent of a
            # i.e., T[a,b,c]*T[a',b0,c0] == T[a',b,c]*T[a,b0,c0] for all a,a',b,c

            ref_a = None
            for a in range(U_rows):
                if u1[a]*v1[b0]*w1[c0] + u2[a]*v2[b0]*w2[c0] != 0:
                    ref_a = a
                    break

            if ref_a is None:
                return True, ([0]*U_rows, [0]*V_rows, [0]*W_rows)

            ref_val = u1[ref_a]*v1[b0]*w1[c0] + u2[ref_a]*v2[b0]*w2[c0]

            # Check separability
            for a in range(U_rows):
                if u1[a]*v1[b0]*w1[c0] + u2[a]*v2[b0]*w2[c0] == 0:
                    # This a must have T[a,b,c] = 0 for all b,c
                    for b, c in bc_pairs:
                        if u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c] != 0:
                            return False, None
                    continue

            for b, c in bc_pairs:
                for a in range(U_rows):
                    t_ac = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
                    t_ref = u1[a]*v1[b0]*w1[c0] + u2[a]*v2[b0]*w2[c0]
                    if t_ref == 0:
                        if t_ac != 0:
                            return False, None
                    else:
                        ratio = t_ac / t_ref
                        # Check that ratio is independent of a
                        for a2 in range(U_rows):
                            t_ac2 = u1[a2]*v1[b]*w1[c] + u2[a2]*v2[b]*w2[c]
                            t_ref2 = u1[a2]*v1[b0]*w1[c0] + u2[a2]*v2[b0]*w2[c0]
                            if t_ref2 != 0:
                                if abs(t_ac2/t_ref2 - ratio) > 1e-9:
                                    return False, None

            # If we get here, the tensor is rank-1. Construct u, v, w.
            u = [0.0] * U_rows
            v = [0.0] * V_rows
            w = [0.0] * W_rows

            # Set w[c0] = 1, then v[b0] = 1, u[a] = T[a,b0,c0]
            w[c0] = 1.0
            v[b0] = 1.0
            for a in range(U_rows):
                u[a] = u1[a]*v1[b0]*w1[c0] + u2[a]*v2[b0]*w2[c0]

            # Determine v[b] for other b
            for b in range(V_rows):
                if b == b0:
                    continue
                # Find a,c where T[a,b,c] and T[a,b0,c] are both nonzero
                found = False
                for a in range(U_rows):
                    if u[a] == 0:
                        continue
                    for c in range(W_rows):
                        t1 = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
                        t2 = u1[a]*v1[b0]*w1[c] + u2[a]*v2[b0]*w2[c]
                        if t2 != 0:
                            v[b] = t1 / t2
                            found = True
                            break
                    if found:
                        break
                if not found:
                    v[b] = 0.0

            # Determine w[c] for other c
            for c in range(W_rows):
                if c == c0:
                    continue
                found = False
                for a in range(U_rows):
                    if u[a] == 0:
                        continue
                    for b in range(V_rows):
                        if v[b] == 0:
                            continue
                        t1 = u1[a]*v1[b]*w1[c] + u2[a]*v2[b]*w2[c]
                        t2 = u1[a]*v1[b]*w1[c0] + u2[a]*v2[b]*w2[c0]
                        if t2 != 0:
                            w[c] = t1 / t2
                            found = True
                            break
                    if found:
                        break
                if not found:
                    w[c] = 0.0

            return True, (u, v, w)

        # Try to find mergeable pairs
        merged = [False] * rank
        new_U_cols = []
        new_V_cols = []
        new_W_cols = []

        for i in range(rank):
            if merged[i]:
                continue
            merged_i = False
            for j in range(i + 1, rank):
                if merged[j]:
                    continue
                can_m, merged_term = can_merge(i, j)
                if can_m:
                    u, v, w = merged_term
                    new_U_cols.append(u)
                    new_V_cols.append(v)
                    new_W_cols.append(w)
                    merged[i] = True
                    merged[j] = True
                    merged_i = True
                    break
            if not merged_i:
                new_U_cols.append(U_cols[i])
                new_V_cols.append(V_cols[i])
                new_W_cols.append(W_cols[i])

        if len(new_U_cols) < rank:
            U_cols = new_U_cols
            V_cols = new_V_cols
            W_cols = new_W_cols
            rank = len(U_cols)

    # Convert to matrix format
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]

    for c in range(rank):
        for r in range(U_rows):
            U[r][c] = U_cols[c][r]
        for r in range(V_rows):
            V[r][c] = V_cols[c][r]
        for r in range(W_rows):
            W[r][c] = W_cols[c][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
