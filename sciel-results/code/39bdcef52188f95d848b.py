import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses branch-and-bound to find the maximum disjoint packing of 2x2x2 Strassen blocks,
    with greedy and local-search fallbacks. Then attempts a final polish to merge
    any remaining schoolbook terms that share structure.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 Strassen blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            indices = frozenset()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices = indices | {(i, j, k)}
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    num_blocks = len(all_blocks)
    num_entries = n * m * p  # 105

    # Entry-to-index mapping
    entry_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_to_idx[(i, j, k)] = idx
                idx += 1

    # Build conflict matrix: conflict[a][b] = True if blocks a and b share an entry
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if all_blocks[a][1] & all_blocks[b][1]:
                conflict[a][b] = conflict[b][a] = True

    # ---- Branch and Bound for maximum independent set (set packing) ----
    best_packing = []
    best_num_blocks = 0

    def branch_and_bound():
        nonlocal best_packing, best_num_blocks

        # Sort blocks by degree (number of conflicts) descending for better pruning
        block_order = sorted(range(num_blocks), key=lambda b: sum(conflict[b]), reverse=True)

        selected = []
        available = set(range(num_blocks))

        # Precompute for each block, the set of blocks it conflicts with
        conflict_sets = [set() for _ in range(num_blocks)]
        for a in range(num_blocks):
            for b in range(a + 1, num_blocks):
                if conflict[a][b]:
                    conflict_sets[a].add(b)
                    conflict_sets[b].add(a)

        def bb(selected, available, depth):
            nonlocal best_packing, best_num_blocks

            if time.time() > deadline - 10.0:
                return

            # Upper bound: current selected + max possible from available
            # Greedy upper bound: repeatedly pick the block with fewest conflicts among available
            ub = len(selected)
            avail = set(available)
            while avail:
                # Pick block with minimum conflicts in avail
                min_conf = None
                min_count = float('inf')
                for b in avail:
                    c = len(conflict_sets[b] & avail)
                    if c < min_count:
                        min_count = c
                        min_conf = b
                if min_conf is None:
                    break
                avail.discard(min_conf)
                avail -= conflict_sets[min_conf]
                ub += 1

            if ub <= best_num_blocks:
                return

            if not available:
                if len(selected) > best_num_blocks:
                    best_num_blocks = len(selected)
                    best_packing = selected[:]
                return

            # Choose the next block to branch on: pick one with highest degree in available
            chosen = max(available, key=lambda b: len(conflict_sets[b] & available))

            # Branch 1: include chosen
            new_selected = selected + [chosen]
            new_available = available - conflict_sets[chosen] - {chosen}
            bb(new_selected, new_available, depth + 1)

            # Branch 2: exclude chosen
            bb(selected, available - {chosen}, depth + 1)

        bb([], set(range(num_blocks)), 0)

    # Run branch and bound
    branch_and_bound()

    # Fallback: greedy with random restarts if B&B didn't find a good solution
    if best_num_blocks < 9 and time.time() < deadline - 20.0:
        max_trials = 5000
        trial = 0
        while trial < max_trials and time.time() < deadline - 20.0:
            trial += 1
            random.shuffle(all_blocks)
            used_indices = set()
            current_packing = []

            for block_params, block_indices in all_blocks:
                if not (block_indices & used_indices):
                    current_packing.append(block_params)
                    used_indices |= block_indices

            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = current_packing[:]
                if best_num_blocks >= 9:
                    break

    # Local search improvement: try to swap blocks to increase count
    if best_num_blocks > 0 and best_num_blocks < 9 and time.time() < deadline - 15.0:
        current_packing = best_packing[:]
        used_indices = set()
        for b in current_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        used_indices.add((i, j, k))

        improved = True
        max_local_iters = 100
        local_iter = 0
        while improved and local_iter < max_local_iters and time.time() < deadline - 15.0:
            improved = False
            local_iter += 1

            for remove_pos in range(len(current_packing)):
                if time.time() > deadline - 15.0:
                    break

                removed_params = current_packing[remove_pos]
                i1, i2, j1, j2, k1, k2 = removed_params
                removed_indices = set()
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            removed_indices.add((i, j, k))

                temp_used = used_indices - removed_indices
                temp_packing = current_packing[:remove_pos] + current_packing[remove_pos+1:]

                candidates = []
                for idx_b in range(num_blocks):
                    block_params, block_indices = all_blocks[idx_b]
                    if not (block_indices & temp_used):
                        candidates.append(idx_b)

                random.shuffle(candidates)

                found_swap = False
                for c1 in candidates[:80]:
                    if found_swap:
                        break
                    b1_indices = all_blocks[c1][1]
                    temp_used_new = temp_used | b1_indices

                    for c2 in candidates[:80]:
                        if c2 == c1:
                            continue
                        b2_indices = all_blocks[c2][1]
                        if not (b2_indices & temp_used_new):
                            current_packing = temp_packing + [all_blocks[c1][0], all_blocks[c2][0]]
                            used_indices = temp_used_new | b2_indices
                            improved = True
                            found_swap = True

                            if len(current_packing) > best_num_blocks:
                                best_num_blocks = len(current_packing)
                                best_packing = current_packing[:]
                            break
                    if found_swap:
                        break
                if improved:
                    break

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

    # Collect uncovered entries
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    uncovered.append((i, j, k))

    # Try to merge pairs of schoolbook terms that share a U or V entry
    # If two terms share the same U entry (i,j), they can potentially be merged
    # using the identity: a_ij*b_jk1 + a_ij*b_jk2 = a_ij*(b_jk1+b_jk2)
    # This saves 1 term if two uncovered entries share the same (i,j) but different k
    # Similarly for shared V entries (j,k) with different i.

    merged_terms = []
    used_uncovered = set()

    # Group uncovered by (i,j) - same U entry
    by_ij = {}
    for (i, j, k) in uncovered:
        key = (i, j)
        if key not in by_ij:
            by_ij[key] = []
        by_ij[key].append(k)

    # For each group with >= 2 k values, merge into one term
    for (i, j), k_list in by_ij.items():
        if len(k_list) >= 2:
            u_entry = i * m + j
            v_entries = {}
            w_entries = {}
            for k in k_list:
                v_entries[j * p + k] = 1
                w_entries[k * n + i] = 1
            merged_terms.append((u_entry, v_entries, w_entries))
            for k in k_list:
                used_uncovered.add((i, j, k))

    # Group remaining uncovered by (j,k) - same V entry
    remaining = [t for t in uncovered if t not in used_uncovered]
    by_jk = {}
    for (i, j, k) in remaining:
        key = (j, k)
        if key not in by_jk:
            by_jk[key] = []
        by_jk[key].append(i)

    for (j, k), i_list in by_jk.items():
        if len(i_list) >= 2:
            v_entry = j * p + k
            u_entries = {}
            w_entries = {}
            for i in i_list:
                u_entries[i * m + j] = 1
                w_entries[k * n + i] = 1
            merged_terms.append((u_entries, v_entry, w_entries))
            for i in i_list:
                used_uncovered.add((i, j, k))

    # Add remaining uncovered as individual schoolbook terms
    for (i, j, k) in uncovered:
        if (i, j, k) not in used_uncovered:
            add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    # Add merged terms
    for term in merged_terms:
        if isinstance(term[0], int):
            # (u_entry, v_entries, w_entries)
            add_term({term[0]: 1}, term[1], term[2])
        else:
            # (u_entries, v_entry, w_entries)
            add_term(term[0], {term[1]: 1}, term[2])

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
