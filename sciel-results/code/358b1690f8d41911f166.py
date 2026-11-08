import time
import random
from itertools import combinations

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a optimized backtracking search to find the maximum number of disjoint
    2x2x2 Strassen blocks, then fills remaining terms with schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 110.0

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
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), indices))
    
    num_blocks = len(all_blocks)
    
    # Build conflict graph
    conflict = [[False] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if all_blocks[i][1] & all_blocks[j][1]:
                conflict[i][j] = True
                conflict[j][i] = True

    # Fast backtracking for Max Independent Set
    best_set = set()
    best_size = 0
    
    # Sort blocks by degree (most constrained first) to prune earlier
    # But we need to maintain indices.
    # Let's just use a simple recursive search with a good bound.
    
    def max_is(candidates, current_set):
        nonlocal best_set, best_size
        nonlocal deadline
        
        if time.time() > deadline - 5.0:
            return
            
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
            
        # Bound: current + remaining
        if len(current_set) + len(candidates) <= best_size:
            return
        
        # Pick a block with max conflicts in candidates
        # This helps prune
        # For simplicity, pick first, but let's try to pick the one with most conflicts
        # Actually, picking the one with fewest conflicts is often better for MIS?
        # Or most?
        # Let's stick to first for now, but ensure we try both include and exclude.
        
        first = candidates[0]
        rest = candidates[1:]
        
        # Option 1: Include first
        new_candidates = [c for c in rest if not conflict[first][c]]
        max_is(new_candidates, current_set | {first})
        
        # Option 2: Exclude first
        max_is(rest, current_set)

    # To speed up, we can initialize best_size with a greedy solution
    greedy_set = set()
    remaining = list(range(num_blocks))
    while remaining:
        # Pick block with min conflicts in remaining
        min_conf_idx = 0
        min_conf_count = num_blocks
        for idx in remaining:
            conf_count = sum(1 for c in remaining if conflict[idx][c])
            if conf_count < min_conf_count:
                min_conf_count = conf_count
                min_conf_idx = idx
        greedy_set.add(min_conf_idx)
        remaining = [c for c in remaining if c != min_conf_idx and not conflict[min_conf_idx][c]]
        
    best_size = len(greedy_set)
    best_set = set(greedy_set)
    
    max_is(list(range(num_blocks)), set())
    
    # Extract the best packing
    best_packing = [all_blocks[i][0] for i in sorted(best_set)]
    best_num_blocks = len(best_packing)
    
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
    for block_params in best_packing:
        i1, i2, j1, j2, k1, k2 = block_params
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    break
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
