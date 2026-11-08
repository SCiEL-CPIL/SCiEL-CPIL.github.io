import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0

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
    
    # Pre-compute compatibility: two blocks are compatible if their index sets are disjoint
    # For speed, use bitmask representation
    # Total indices = 3*5*7 = 105, too many for a single int bitmask in Python efficiently
    # But we can use Python's arbitrary precision integers
    
    # Map each (i,j,k) to a bit position
    idx_to_bit = {}
    bit_to_idx = {}
    bit = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_to_bit[(i, j, k)] = bit
                bit_to_idx[bit] = (i, j, k)
                bit += 1
    
    block_bitmasks = []
    for params, indices, _ in all_blocks:
        mask = 0
        for idx in indices:
            mask |= (1 << idx_to_bit[idx])
        block_bitmasks.append(mask)
    
    # Compatibility matrix: compatible[a][b] = True if blocks a and b are disjoint
    # Two blocks are disjoint iff (mask_a & mask_b) == 0
    # Since all blocks use 8 indices, and total is 105, max blocks = floor(105/8) = 13
    
    best_num_blocks = 0
    best_packing = []
    
    # Greedy with multiple restarts and different ordering strategies
    num_trials = 3000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        
        # Different ordering strategies
        if t % 3 == 0:
            order = list(range(num_blocks_total))
            random.shuffle(order)
        elif t % 3 == 1:
            # Sort by number of compatible blocks (most constrained first)
            # Pre-compute degree
            if not hasattr(construct, '_degrees'):
                construct._degrees = []
                for a in range(num_blocks_total):
                    deg = 0
                    for b in range(num_blocks_total):
                        if a != b and (block_bitmasks[a] & block_bitmasks[b]) == 0:
                            deg += 1
                    construct._degrees.append(deg)
            order = sorted(range(num_blocks_total), key=lambda x: construct._degrees[x])
            random.shuffle(order)
        else:
            # Weighted random: prefer blocks that cover "rare" indices
            order = list(range(num_blocks_total))
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
            if best_num_blocks >= 13:
                break
    
    # Local improvement: try to swap blocks to improve packing
    if time.time() < deadline - 15.0 and best_num_blocks < 13:
        improved = True
        while improved:
            improved = False
            current_set = set(best_packing)
            used_mask = 0
            for idx in best_packing:
                used_mask |= block_bitmasks[idx]
            
            for i in range(len(best_packing)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(num_blocks_total):
                    if j in current_set:
                        continue
                    # Try replacing best_packing[i] with j
                    if (block_bitmasks[j] & used_mask) == 0:
                        # j is compatible with all others except possibly best_packing[i]
                        new_used = used_mask ^ block_bitmasks[best_packing[i]] | block_bitmasks[j]
                        # Check j is compatible with all other blocks in packing
                        ok = True
                        for k in range(len(best_packing)):
                            if k == i:
                                continue
                            if (block_bitmasks[j] & block_bitmasks[best_packing[k]]) != 0:
                                ok = False
                                break
                        if ok:
                            best_packing[i] = j
                            current_set.remove(best_packing[i])
                            current_set.add(j)
                            used_mask = new_used
                            improved = True
    
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
