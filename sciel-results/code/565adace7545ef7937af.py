import time
import random
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices, frozenset(indices)))

    num_blocks_total = len(all_blocks)
    idx_count = n * m * p

    def idx_to_id(i, j, k):
        return i * m * p + j * p + k

    # Build constraint matrix A
    A = np.zeros((idx_count, num_blocks_total), dtype=int)
    for b_idx, (params, indices, _) in enumerate(all_blocks):
        for idx in indices:
            i, j, k = idx
            r = idx_to_id(i, j, k)
            A[r, b_idx] = 1

    best_packing = []
    best_num_blocks = 0

    # Try ILP
    try:
        c = -np.ones(num_blocks_total)
        bounds = [(0, 1)] * num_blocks_total
        integrality = np.ones(num_blocks_total, dtype=int)
        
        # Use a reasonable time limit for the solver
        res = scipy.optimize.milp(
            c,
            constraints=scipy.optimize.LinearConstraint(A, ub=1),
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": 100, "mip_gap": 0.0}
        )
        
        if res.status in [0, 3]:
            x = np.round(res.x)
            for b_idx in range(num_blocks_total):
                if x[b_idx] > 0.5:
                    best_packing.append(b_idx)
            best_num_blocks = len(best_packing)
    except Exception:
        pass

    # Fallback Heuristic if ILP failed or found few blocks
    if best_num_blocks < 7:
        # Prepare bitmasks for heuristic
        block_bitmasks = []
        for params, indices, _ in all_blocks:
            mask = 0
            for idx in indices:
                i, j, k = idx
                bit = idx_to_id(i, j, k)
                mask |= (1 << bit)
            block_bitmasks.append(mask)
            
        # Precompute conflicts and degrees for better heuristic
        conflicts = [[] for _ in range(num_blocks_total)]
        for a in range(num_blocks_total):
            for b in range(a + 1, num_blocks_total):
                if (block_bitmasks[a] & block_bitmasks[b]) != 0:
                    conflicts[a].append(b)
                    conflicts[b].append(a)
        degrees = [len(conflicts[a]) for a in range(num_blocks_total)]

        # Run several trials of greedy with different orderings
        num_trials = 2000
        for t in range(num_trials):
            if time.time() > deadline - 20.0:
                break
            
            if t % 2 == 0:
                order = list(range(num_blocks_total))
                random.shuffle(order)
            else:
                # Most constrained first
                order = sorted(range(num_blocks_total), key=lambda x: degrees[x])
                random.shuffle(order)

            used_mask = 0
            current_packing = []
            for idx in order:
                if (block_bitmasks[idx] & used_mask) == 0:
                    current_packing.append(idx)
                    used_mask |= block_bitmasks[idx]
            
            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = current_packing[:]

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

        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})

    covered = set()
    for block_idx in best_packing:
        params, indices, _ = all_blocks[block_idx]
        i1, i2, j1, j2, k1, k2 = params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered |= set(indices)

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})

    rank = len(U_cols)

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
