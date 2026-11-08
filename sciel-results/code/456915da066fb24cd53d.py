import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses ILP (via scipy) to find maximum disjoint packing of 2x2x2 Strassen blocks.
    Falls back to greedy search if ILP fails.
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
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))

    num_blocks = len(all_blocks)
    num_entries = n * m * p  # 105

    # Create entry-to-index mapping
    entry_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                entry_to_idx[(i, j, k)] = idx
                idx += 1

    # Build constraint matrix: A[entry_idx, block_idx] = 1 if block covers entry
    A = np.zeros((num_entries, num_blocks), dtype=np.int8)
    for b_idx, (params, indices) in enumerate(all_blocks):
        for e in indices:
            A[entry_to_idx[e], b_idx] = 1

    # Try ILP via scipy
    best_packing = []
    best_num_blocks = 0

    try:
        from scipy.optimize import linprog

        # Maximize sum(x_b) subject to A @ x <= 1, 0 <= x <= 1, x binary
        # linprog minimizes, so minimize -sum(x_b)
        c = -np.ones(num_blocks)
        A_ub = A
        b_ub = np.ones(num_entries)
        bounds = [(0, 1)] * num_blocks

        # Use high-resolution method for integer solution
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            # Round the solution
            x = np.round(res.x)
            x = np.clip(x, 0, 1)
            selected = np.where(x > 0.5)[0]
            if len(selected) > best_num_blocks:
                best_num_blocks = len(selected)
                best_packing = [all_blocks[i][0] for i in selected]
    except Exception:
        pass

    # Fallback: greedy search with many restarts if ILP didn't find good solution
    if best_num_blocks < 9 and time.time() < deadline - 20.0:
        max_trials = 3000
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
        max_local_iters = 50
        local_iter = 0
        while improved and local_iter < max_local_iters and time.time() < deadline - 15.0:
            improved = False
            local_iter += 1

            # Try removing one block and adding two non-conflicting blocks
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

                # Find candidates that don't conflict
                candidates = []
                for idx in range(num_blocks):
                    block_params, block_indices = all_blocks[idx]
                    if not (block_indices & temp_used):
                        candidates.append(idx)

                random.shuffle(candidates)

                found_swap = False
                for c1 in candidates[:50]:
                    if found_swap:
                        break
                    b1_indices = all_blocks[c1][1]
                    temp_used_new = temp_used | b1_indices

                    for c2 in candidates[:50]:
                        if c2 == c1:
                            continue
                        b2_indices = all_blocks[c2][1]
                        if not (b2_indices & temp_used_new):
                            # Valid swap
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
