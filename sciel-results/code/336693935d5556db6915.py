import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    total_indices = n * m * p  # 105
    
    def idx(i, j, k):
        return (i * m + j) * p + k
    
    # Enumerate all possible 2x2x2 blocks
    blocks = []
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
                                        mask |= (1 << idx(i, j, k))
                            blocks.append((i1, i2, j1, j2, k1, k2, mask))
    
    num_blocks = len(blocks)
    masks = [b[6] for b in blocks]
    
    # Precompute: for each block, which other blocks conflict with it
    # conflict_mask[b] is a bitmask of blocks that conflict with block b
    # But num_blocks can be up to 630, so we use a set-based approach
    # Actually, let's precompute compatible lists
    compatible = [[] for _ in range(num_blocks)]
    for a in range(num_blocks):
        for b in range(a + 1, num_blocks):
            if masks[a] & masks[b] == 0:
                compatible[a].append(b)
                compatible[b].append(a)
    
    # Greedy with a given order
    def greedy(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        used = 0
        selected = []
        for bidx in order:
            if bidx in selected_set:
                continue
            if masks[bidx] & used == 0:
                selected.append(bidx)
                selected_set.add(bidx)
                used |= masks[bidx]
        return selected, used, selected_set
    
    # Fast local search using sets for O(1) membership
    def local_search(selected, used, selected_set, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1) add any fitting block from compatible list
            for bidx in range(num_blocks):
                if bidx in selected_set:
                    continue
                if masks[bidx] & used == 0:
                    selected.append(bidx)
                    selected_set.add(bidx)
                    used |= masks[bidx]
                    improved = True
            # 2) 1-for-2 swaps: remove s, try to add 2 blocks that fit
            if time.time() > time_limit:
                break
            for s in list(selected):
                if time.time() > time_limit:
                    break
                new_used = used & ~masks[s]
                # Find up to 2 blocks that fit in new_used
                added = []
                for bidx in range(num_blocks):
                    if bidx == s or bidx in selected_set:
                        continue
                    if masks[bidx] & new_used == 0:
                        added.append(bidx)
                        new_used |= masks[bidx]
                        if len(added) >= 2:
                            break
                if len(added) >= 2:
                    selected.remove(s)
                    selected_set.discard(s)
                    for a in added:
                        selected.append(a)
                        selected_set.add(a)
                    used = new_used
                    improved = True
            # 3) 2-for-3 swaps
            if time.time() > time_limit:
                break
            sel_list = list(selected)
            for i1 in range(len(sel_list)):
                if time.time() > time_limit:
                    break
                s1 = sel_list[i1]
                for i2 in range(i1 + 1, len(sel_list)):
                    if time.time() > time_limit:
                        break
                    s2 = sel_list[i2]
                    new_used = used & ~masks[s1] & ~masks[s2]
                    added = []
                    for bidx in range(num_blocks):
                        if bidx == s1 or bidx == s2 or bidx in selected_set:
                            continue
                        if masks[bidx] & new_used == 0:
                            added.append(bidx)
                            new_used |= masks[bidx]
                            if len(added) >= 3:
                                break
                    if len(added) >= 3:
                        selected.remove(s1)
                        selected.remove(s2)
                        selected_set.discard(s1)
                        selected_set.discard(s2)
                        for a in added:
                            selected.append(a)
                            selected_set.add(a)
                        used = new_used
                        improved = True
        return selected, used, selected_set
    
    best = []
    best_len = 0
    best_set = set()
    rng = random.Random(12345)
    limit = deadline - 10.0
    
    # Multiple random greedy + local search
    # Use time budget per trial to allow more iterations
    num_trials = 20000
    for t in range(num_trials):
        if time.time() > limit:
            break
        # Random ordering with slight bias toward less-conflicting blocks
        order = list(range(num_blocks))
        rng.shuffle(order)
        selected, used, selected_set = greedy(order)
        selected, used, selected_set = local_search(selected, used, selected_set, limit)
        if len(selected) > best_len:
            best_len = len(selected)
            best = list(selected)
            best_set = set(selected)
            if best_len >= 15:
                break
    
    # Also try "least conflict first" greedy
    conflict_counts = [len(compatible[a]) for a in range(num_blocks)]
    # Fewer compatible = more conflicts = should be placed later? 
    # Actually, we want blocks with fewer conflicts to go first
    order = sorted(range(num_blocks), key=lambda x: conflict_counts[x], reverse=True)
    selected, used, selected_set = greedy(order)
    selected, used, selected_set = local_search(selected, used, selected_set, limit)
    if len(selected) > best_len:
        best_len = len(selected)
        best = list(selected)
        best_set = set(selected)
    
    # Try "most isolated first" - blocks that conflict with fewest others
    order2 = sorted(range(num_blocks), key=lambda x: conflict_counts[x])
    selected, used, selected_set = greedy(order2)
    selected, used, selected_set = local_search(selected, used, selected_set, limit)
    if len(selected) > best_len:
        best_len = len(selected)
        best = list(selected)
        best_set = set(selected)
    
    # Build U, V, W
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        u = [0] * U_rows
        v = [0] * V_rows
        w = [0] * W_rows
        for r, val in u_entries.items():
            u[r] = val
        for r, val in v_entries.items():
            v[r] = val
        for r, val in w_entries.items():
            w[r] = val
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
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
    
    covered = 0
    for bidx in best:
        i1, i2, j1, j2, k1, k2, mask = blocks[bidx]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        covered |= mask
    
    # Schoolbook for uncovered
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
                if not (covered >> idx(i, j, k)) & 1:
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
