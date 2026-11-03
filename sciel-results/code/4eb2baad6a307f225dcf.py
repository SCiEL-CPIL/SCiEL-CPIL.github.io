import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a fast greedy+local search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 105 - num_blocks.
    
    Key improvements:
    1. Precompute all blocks with bitmasks
    2. Fast greedy with multiple random restarts
    3. Efficient local search using precomputed conflict sets
    4. Backtracking search for optimal packing
    """
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    total_triples = n * m * p  # 105

    def triple_to_idx(i, j, k):
        return i * m * p + j * p + k

    # Precompute all possible 2x2x2 blocks
    all_blocks = []
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = []
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.append(idx)
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, tuple(indices)))

    num_all_blocks = len(all_blocks)
    
    # Precompute conflict bitmasks: for each block, which other blocks conflict with it
    # Use a list of sets for fast lookup
    block_conflicts = [set() for _ in range(num_all_blocks)]
    for a in range(num_all_blocks):
        for b in range(a + 1, num_all_blocks):
            if (all_blocks[a][1] & all_blocks[b][1]) != 0:
                block_conflicts[a].add(b)
                block_conflicts[b].add(a)

    # Count how many blocks cover each triple (for rarity heuristic)
    triple_block_count = [0] * total_triples
    for _, _, indices in all_blocks:
        for idx in indices:
            triple_block_count[idx] += 1

    best_num_blocks = 0
    best_packing = []

    def greedy_random():
        """Greedy selection with random ordering."""
        order = list(range(num_all_blocks))
        random.shuffle(order)
        used_mask = 0
        packing = []
        for idx in order:
            bm = all_blocks[idx][1]
            if (bm & used_mask) == 0:
                packing.append(idx)
                used_mask |= bm
        return packing

    def greedy_rarity():
        """Greedy selection prioritizing blocks that cover rare indices."""
        order = list(range(num_all_blocks))
        random.shuffle(order)
        used_mask = 0
        packing = []
        # Sort by rarity score computed dynamically
        scored = []
        for idx in order:
            _, bm, indices = all_blocks[idx]
            if (bm & used_mask) == 0:
                score = sum(1.0 / triple_block_count[i] for i in indices)
                scored.append((score, idx))
        scored.sort(reverse=True)
        for score, idx in scored:
            bm = all_blocks[idx][1]
            if (bm & used_mask) == 0:
                packing.append(idx)
                used_mask |= bm
        return packing

    def local_search(packing):
        """Try to improve packing by 1-for-1 swaps."""
        if not packing:
            return packing
        
        current = list(packing)
        used_mask = 0
        for idx in current:
            used_mask |= all_blocks[idx][1]
        
        improved = True
        while improved:
            if time.time() > deadline - 3.0:
                break
            improved = False
            
            for remove_idx in range(len(current)):
                # Compute mask without this block
                mask_without = used_mask ^ all_blocks[current[remove_idx]][1]
                
                # Try adding a non-conflicting block
                for add_idx in range(num_all_blocks):
                    if add_idx in current:
                        continue
                    if (all_blocks[add_idx][1] & mask_without) == 0:
                        # Check it doesn't conflict with any remaining
                        # mask_without already excludes the removed block
                        new_packing = current[:remove_idx] + current[remove_idx+1:] + [add_idx]
                        current = new_packing
                        used_mask = mask_without | all_blocks[add_idx][1]
                        improved = True
                        break
                if improved:
                    break
        
        return current

    def backtracking_search(available, current_packing, depth=0):
        """Backtracking search for maximum packing."""
        nonlocal best_num_blocks, best_packing
        
        if time.time() > deadline - 2.0:
            return
        
        if len(current_packing) + len(available) <= best_num_blocks:
            return
        
        if not available:
            if len(current_packing) > best_num_blocks:
                best_num_blocks = len(current_packing)
                best_packing = list(current_packing)
            return
        
        # Try each available block
        for i, idx in enumerate(available):
            bm = all_blocks[idx][1]
            # Filter: only blocks that don't conflict with idx
            new_available = []
            for j in range(i + 1, len(available)):
                other_idx = available[j]
                if (bm & all_blocks[other_idx][1]) == 0:
                    new_available.append(other_idx)
            
            current_packing.append(idx)
            backtracking_search(new_available, current_packing, depth + 1)
            current_packing.pop()

    # Phase 1: Many random greedy trials
    num_trials = 5000
    for t in range(num_trials):
        if time.time() > deadline - 15.0:
            break
        
        if t % 2 == 0:
            packing = greedy_random()
        else:
            packing = greedy_rarity()
        
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            if best_num_blocks >= 10:
                break

    # Phase 2: Local search on best packing
    if best_packing and time.time() < deadline - 10.0:
        improved = local_search(best_packing)
        if len(improved) > best_num_blocks:
            best_num_blocks = len(improved)
            best_packing = improved

    # Phase 3: Try local search on several random packings
    for t in range(200):
        if time.time() > deadline - 5.0:
            break
        packing = greedy_random()
        if len(packing) >= best_num_blocks - 1:
            improved = local_search(packing)
            if len(improved) > best_num_blocks:
                best_num_blocks = len(improved)
                best_packing = improved
                if best_num_blocks >= 10:
                    break

    # Phase 4: Backtracking search from a good starting point
    if time.time() < deadline - 5.0 and best_num_blocks < 10:
        # Start from the best packing, try to add more via backtracking
        # Build available set from all blocks not in best_packing
        in_packing = set(best_packing)
        available = [i for i in range(num_all_blocks) if i not in in_packing]
        
        # Filter available to only those compatible with current packing
        current_mask = 0
        for idx in best_packing:
            current_mask |= all_blocks[idx][1]
        available = [i for i in available if (all_blocks[i][1] & current_mask) == 0]
        
        # Sort available by rarity
        available.sort(key=lambda i: -sum(1.0 / triple_block_count[j] for j in all_blocks[i][2]))
        
        # Try to add blocks via backtracking (limited depth)
        def bt_add(available, current_packing, max_depth=3):
            nonlocal best_num_blocks, best_packing
            if time.time() > deadline - 2.0 or max_depth <= 0:
                return
            if len(current_packing) + len(available) <= best_num_blocks:
                return
            if not available:
                if len(current_packing) > best_num_blocks:
                    best_num_blocks = len(current_packing)
                    best_packing = list(current_packing)
                return
            
            for i, idx in enumerate(available[:30]):  # Limit branching
                bm = all_blocks[idx][1]
                new_available = []
                for j in range(i + 1, len(available)):
                    other_idx = available[j]
                    if (bm & all_blocks[other_idx][1]) == 0:
                        new_available.append(other_idx)
                
                current_packing.append(idx)
                bt_add(new_available, current_packing, max_depth - 1)
                current_packing.pop()
        
        bt_add(available, list(best_packing), max_depth=4)

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
    for block_idx in best_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx][0]
        add_strassen_block(i1, i2, j1, j2, k1, k2)
        for i in (i1, i2):
            for j in (j1, j2):
                for k in (k1, k2):
                    covered.add((i, j, k))

    # Add schoolbook terms for uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
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
