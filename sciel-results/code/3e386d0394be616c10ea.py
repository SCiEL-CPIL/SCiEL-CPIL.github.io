import time
import random
import itertools
from collections import defaultdict

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a combination of Strassen block packing and local search to minimize rank.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_cells = n * m * p  # 105

    # Generate all possible 2x2x2 Strassen blocks with bitmask representation
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            mask = 0
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        cell_idx = (i * m + j) * p + k
                                        mask |= (1 << cell_idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), mask))

    num_blocks_total = len(all_blocks)

    # Precompute conflict graph
    conflict = [set() for _ in range(num_blocks_total)]
    for i in range(num_blocks_total):
        for j in range(i + 1, num_blocks_total):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i].add(j)
                conflict[j].add(i)

    block_order = sorted(range(num_blocks_total), key=lambda x: len(conflict[x]), reverse=True)

    best_num_blocks = 0
    best_packing = []

    # Branch and bound for maximum independent set (set packing)
    def branch_and_bound(node_idx, used_mask, current_packing, current_count):
        nonlocal best_num_blocks, best_packing, deadline

        if time.time() > deadline - 5.0:
            return

        remaining = num_blocks_total - node_idx
        if current_count + remaining <= best_num_blocks:
            return

        if node_idx >= num_blocks_total:
            if current_count > best_num_blocks:
                best_num_blocks = current_count
                best_packing = current_packing[:]
            return

        block_idx = block_order[node_idx]
        params, mask = all_blocks[block_idx]

        # Option 1: Skip this block
        branch_and_bound(node_idx + 1, used_mask, current_packing, current_count)

        # Option 2: Include this block if it doesn't conflict
        if (used_mask & mask) == 0:
            branch_and_bound(node_idx + 1, used_mask | mask, current_packing + [params], current_count + 1)

    branch_and_bound(0, 0, [], 0)

    # Greedy fallback
    if best_num_blocks < 5:
        num_trials = 5000
        for t in range(num_trials):
            if time.time() > deadline - 20.0:
                break
            order = list(range(num_blocks_total))
            random.shuffle(order)
            used_mask = 0
            current_packing = []

            for idx in order:
                params, mask = all_blocks[idx]
                if (used_mask & mask) == 0:
                    current_packing.append(params)
                    used_mask |= mask

            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = current_packing[:]
                if best_num_blocks >= 12:
                    break

    # Local search
    if time.time() < deadline - 10.0:
        current_packing = best_packing[:]
        used_mask = 0
        for b in current_packing:
            i1, i2, j1, j2, k1, k2 = b
            for i in (i1, i2):
                for j in (j1, j2):
                    for k in (k1, k2):
                        cell_idx = (i * m + j) * p + k
                        used_mask |= (1 << cell_idx)

        improved = True
        max_local_iters = 100
        for iter_num in range(max_local_iters):
            if time.time() > deadline - 8.0:
                break
            improved = False

            for idx_remove in range(len(current_packing)):
                if time.time() > deadline - 8.0:
                    break

                removed_block = current_packing[idx_remove]
                i1, i2, j1, j2, k1, k2 = removed_block
                removed_mask = 0
                for i in (i1, i2):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            cell_idx = (i * m + j) * p + k
                            removed_mask |= (1 << cell_idx)

                temp_used = used_mask & ~removed_mask
                temp_packing = current_packing[:idx_remove] + current_packing[idx_remove+1:]
                temp_params = set(temp_packing)

                candidates = []
                for idx in range(num_blocks_total):
                    if all_blocks[idx][0] in temp_params:
                        continue
                    if (all_blocks[idx][1] & temp_used) == 0:
                        candidates.append(idx)

                random.shuffle(candidates)

                found = False
                for c1 in candidates[:50]:
                    if found or time.time() > deadline - 8.0:
                        break
                    p1, m1 = all_blocks[c1]
                    new_used = temp_used | m1
                    for c2 in candidates[:50]:
                        if c2 == c1:
                            continue
                        p2, m2 = all_blocks[c2]
                        if (m2 & new_used) == 0:
                            new_packing = temp_packing + [p1, p2]
                            new_used_mask = new_used | m2
                            current_packing = new_packing
                            used_mask = new_used_mask
                            improved = True
                            found = True
                            if len(current_packing) > best_num_blocks:
                                best_num_blocks = len(current_packing)
                                best_packing = current_packing[:]
                            break
                if found:
                    break

            if not improved:
                break

    if best_num_blocks > 0:
        final_packing = best_packing
    else:
        final_packing = []

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
    for block_params in final_packing:
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

    # Local search: try to reduce rank by merging terms
    if time.time() < deadline - 5.0:
        # Try to find pairs of terms that can be merged
        # Two terms (u1,v1,w1) and (u2,v2,w2) can be merged if there exists
        # (u,v,w) such that u1*v1*w1 + u2*v2*w2 = u*v*w
        # This is a complex condition. Instead, try simple heuristics.

        # Heuristic: if two terms have the same support pattern in U, V, W,
        # they might be mergeable.
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break

            # Try random pairs
            r1, r2 = random.sample(range(len(U_cols)), 2)
            # Check if they can be merged (simplified check)
            # For now, just try to see if removing one and adjusting the other works
            pass

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
