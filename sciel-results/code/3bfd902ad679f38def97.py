import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses greedy set-packing with efficient local search to maximize disjoint 2x2x2 Strassen blocks.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS
    U_rows, V_rows, W_rows = n * m, m * p, p * n

    # Precompute all blocks with their index sets (frozen for speed)
    all_blocks = []
    block_index_cache = {}
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            params = (i1, i2, j1, j2, k1, k2)
                            idx_set = frozenset((i, j, k) for i in (i1, i2) for j in (j1, j2) for k in (k1, k2))
                            all_blocks.append((params, idx_set))
                            block_index_cache[params] = idx_set

    num_blocks = len(all_blocks)

    def get_used(packing):
        used = set()
        for b in packing:
            used |= block_index_cache[b]
        return used

    # Phase 1: Greedy with multiple strategies
    best_num_blocks = 0
    best_packing = []

    # Deterministic orderings
    orderings = [
        all_blocks,
        sorted(all_blocks, key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4], x[0][5])),
        sorted(all_blocks, key=lambda x: (x[0][2], x[0][3], x[0][0], x[0][1], x[0][4], x[0][5])),
        sorted(all_blocks, key=lambda x: (x[0][4], x[0][5], x[0][0], x[0][1], x[0][2], x[0][3])),
        sorted(all_blocks, key=lambda x: (x[0][0], x[0][2], x[0][4], x[0][1], x[0][3], x[0][5])),
    ]

    for ordering in orderings:
        if time.time() > deadline - 60.0:
            break
        used = set()
        packing = []
        for params, idx_set in ordering:
            if not (idx_set & used):
                packing.append(params)
                used |= idx_set
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]

    # Random restarts with adaptive budget
    remaining = deadline - time.time()
    num_trials = int(min(10000, max(3000, remaining * 200)))
    for t in range(num_trials):
        if time.time() > deadline - 30.0:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        used = set()
        packing = []
        for params, idx_set in shuffled:
            if not (idx_set & used):
                packing.append(params)
                used |= idx_set
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing[:]
            if best_num_blocks >= 10:
                break

    # Phase 2: Efficient 1-for-2 local search with candidate filtering
    if time.time() < deadline - 5.0:
        current = best_packing[:]
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            used = get_used(current)
            current_set = set(current)

            for idx_remove in range(len(current)):
                if time.time() > deadline - 5.0:
                    break
                rem_block = current[idx_remove]
                rem_idx = block_index_cache[rem_block]
                temp_used = used - rem_idx
                temp_set = current_set - {rem_block}

                # Find candidate blocks that fit in temp_used
                candidates = []
                for b_idx in range(num_blocks):
                    bp, bidx = all_blocks[b_idx]
                    if bp in temp_set:
                        continue
                    if bidx & temp_used:
                        continue
                    candidates.append(b_idx)

                # Try pairs from candidates
                n_cand = len(candidates)
                best_pair = None
                for a in range(n_cand):
                    if time.time() > deadline - 5.0:
                        break
                    ca = candidates[a]
                    pa, ia = all_blocks[ca]
                    tu_a = temp_used | ia
                    for b in range(a + 1, n_cand):
                        cb = candidates[b]
                        pb, ib = all_blocks[cb]
                        if ib & tu_a:
                            continue
                        best_pair = (ca, cb)
                        break
                    if best_pair:
                        break

                if best_pair:
                    ca, cb = best_pair
                    current.pop(idx_remove)
                    current.append(all_blocks[ca][0])
                    current.append(all_blocks[cb][0])
                    improved = True
                    if len(current) > best_num_blocks:
                        best_num_blocks = len(current)
                        best_packing = current[:]
                    break  # Restart scan after modification

    # Phase 3: 2-for-3 swaps with tight candidate filtering and early exit
    if time.time() < deadline - 3.0 and best_num_blocks < 10:
        current = best_packing[:]
        improved = True
        max_swaps = 5  # Limit number of swap attempts to avoid excessive computation
        swap_count = 0
        while improved and time.time() < deadline - 3.0 and swap_count < max_swaps:
            improved = False
            used = get_used(current)
            current_set = set(current)

            found_swap = None
            for i1 in range(len(current)):
                if time.time() > deadline - 3.0:
                    break
                for i2 in range(i1 + 1, len(current)):
                    if time.time() > deadline - 3.0:
                        break
                    r1, r2 = current[i1], current[i2]
                    rem_idx = block_index_cache[r1] | block_index_cache[r2]
                    temp_used = used - rem_idx
                    temp_set = current_set - {r1, r2}

                    candidates = []
                    for b_idx in range(num_blocks):
                        bp, bidx = all_blocks[b_idx]
                        if bp in temp_set:
                            continue
                        if bidx & temp_used:
                            continue
                        candidates.append(b_idx)

                    n_cand = len(candidates)
                    if n_cand < 3:
                        continue
                    triple = None
                    for a in range(n_cand):
                        if time.time() > deadline - 3.0:
                            break
                        ca = candidates[a]
                        ia = all_blocks[ca][1]
                        tu_a = temp_used | ia
                        for b in range(a + 1, n_cand):
                            cb = candidates[b]
                            ib = all_blocks[cb][1]
                            if ib & tu_a:
                                continue
                            tu_ab = tu_a | ib
                            for c in range(b + 1, n_cand):
                                cc = candidates[c]
                                ic = all_blocks[cc][1]
                                if ic & tu_ab:
                                    continue
                                triple = (ca, cb, cc)
                                break
                            if triple:
                                break
                        if triple:
                            break

                    if triple:
                        found_swap = (i1, i2, triple)
                        break
                if found_swap:
                    break

            if found_swap:
                i1, i2, (ca, cb, cc) = found_swap
                hi = max(i1, i2)
                lo = min(i1, i2)
                current.pop(hi)
                current.pop(lo)
                current.append(all_blocks[ca][0])
                current.append(all_blocks[cb][0])
                current.append(all_blocks[cc][0])
                improved = True
                swap_count += 1
                if len(current) > best_num_blocks:
                    best_num_blocks = len(current)
                    best_packing = current[:]

    # Build U, V, W
    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_e, v_e, w_e):
        U_cols.append([0] * U_rows)
        V_cols.append([0] * V_rows)
        W_cols.append([0] * W_rows)
        for idx, val in u_e.items():
            U_cols[-1][idx] = val
        for idx, val in v_e.items():
            V_cols[-1][idx] = val
        for idx, val in w_e.items():
            W_cols[-1][idx] = val

    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        a11, a12 = i1 * m + j1, i1 * m + j2
        a21, a22 = i2 * m + j1, i2 * m + j2
        b11, b12 = j1 * p + k1, j1 * p + k2
        b21, b22 = j2 * p + k1, j2 * p + k2
        c11, c12 = k1 * n + i1, k2 * n + i1
        c21, c22 = k1 * n + i2, k2 * n + i2

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for bp in best_packing:
        add_strassen_block(*bp)
        for i in (bp[0], bp[1]):
            for j in (bp[2], bp[3]):
                for k in (bp[4], bp[5]):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)
    U = [[U_cols[c][r] for c in range(rank)] for r in range(U_rows)]
    V = [[V_cols[c][r] for c in range(rank)] for r in range(V_rows)]
    W = [[W_cols[c][r] for c in range(rank)] for r in range(W_rows)]

    return {"U": U, "V": V, "W": W}

    # Construct U, V, W based on the best packing found
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

    # Add the Strassen blocks from the best packing
    covered = set()
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

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
