import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21
    
    # Precompute all possible 2x2x2 blocks
    # A block is defined by (i1, i2, j1, j2, k1, k2) with i1 < i2, j1 < j2, k1 < k2
    all_blocks = []
    triple_to_idx = lambda i, j, k: i * m * p + j * p + k
    
    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for j1 in range(m - 1):
                for j2 in range(j1 + 1, m):
                    for k1 in range(p - 1):
                        for k2 in range(k1 + 1, p):
                            bitmask = 0
                            indices = set()
                            for i in (i1, i2):
                                for j in (j1, j2):
                                    for k in (k1, k2):
                                        idx = triple_to_idx(i, j, k)
                                        bitmask |= (1 << idx)
                                        indices.add((i, j, k))
                            all_blocks.append(((i1, i2, j1, j2, k1, k2), bitmask, indices))
    
    num_blocks_list = len(all_blocks)
    
    # Count how many blocks cover each triple (for rarity heuristic)
    triple_block_count = [0] * (n * m * p)
    for _, _, indices in all_blocks:
        for i, j, k in indices:
            triple_block_count[triple_to_idx(i, j, k)] += 1
            
    # Compute pairwise conflict counts for sorting
    # conflict[i] = number of blocks j such that block i and block j overlap
    conflict_counts = [0] * num_blocks_list
    for i in range(num_blocks_list):
        bm_i = all_blocks[i][1]
        for j in range(i + 1, num_blocks_list):
            bm_j = all_blocks[j][1]
            if (bm_i & bm_j) != 0:
                conflict_counts[i] += 1
                conflict_counts[j] += 1
                
    # Sort indices by conflict count (ascending) - less conflicting blocks first
    sorted_indices = sorted(range(num_blocks_list), key=lambda x: conflict_counts[x])
    
    best_num_blocks = 0
    best_packing = []
    
    def evaluate_packing(packing):
        return len(packing)
        
    # Heuristic 1: Greedy with conflict-sorted order
    def greedy_conflict():
        used_mask = 0
        packing = []
        for idx in sorted_indices:
            params, bm, _ = all_blocks[idx]
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm
        return packing

    # Heuristic 2: Greedy with random shuffle
    def greedy_random(rng):
        used_mask = 0
        packing = []
        indices = list(range(num_blocks_list))
        rng.shuffle(indices)
        for idx in indices:
            params, bm, _ = all_blocks[idx]
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm
        return packing

    # Heuristic 3: Greedy with rarity priority
    def greedy_rarity(rng):
        used_mask = 0
        packing = []
        indices = list(range(num_blocks_list))
        rng.shuffle(indices)
        
        # Calculate dynamic priority based on uncovered rarity
        # To keep it fast, we just sort once by static rarity or use a simple priority queue simulation
        # Here we use a simple iterative selection: pick the block with max rarity score among available
        available = set(range(num_blocks_list))
        
        # Precompute static rarity score for each block
        static_rarity = [0.0] * num_blocks_list
        for idx in range(num_blocks_list):
            _, _, indices_set = all_blocks[idx]
            score = 0.0
            for i, j, k in indices_set:
                t_idx = triple_to_idx(i, j, k)
                if triple_block_count[t_idx] > 0:
                    score += 1.0 / triple_block_count[t_idx]
            static_rarity[idx] = score
            
        # Sort by static rarity descending
        sorted_by_rarity = sorted(available, key=lambda x: static_rarity[x], reverse=True)
        
        for idx in sorted_by_rarity:
            if idx not in available:
                continue
            params, bm, _ = all_blocks[idx]
            if (bm & used_mask) == 0:
                packing.append(params)
                used_mask |= bm
                available.remove(idx)
                
        return packing

    # Local Search: Try to improve a packing by swapping 1 out, 2 in, or 2 out, 3 in, etc.
    # We focus on 1->2 swaps (remove 1 block, add 2 disjoint blocks that don't conflict with the rest)
    def local_search(packing, rng, max_iters=50):
        current_packing = list(packing)
        current_mask = 0
        for params in current_packing:
            idx = None
            # Find index in all_blocks
            for i, (p, _, _) in enumerate(all_blocks):
                if p == params:
                    idx = i
                    break
            if idx is not None:
                current_mask |= all_blocks[idx][1]
                
        current_size = len(current_packing)
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            improved = False
            # Try removing one block
            for i in range(len(current_packing)):
                if time.time() > deadline - 10.0:
                    break
                    
                removed_params = current_packing[i]
                # Find its index and mask
                removed_idx = None
                removed_mask = 0
                for j, (p, bm, _) in enumerate(all_blocks):
                    if p == removed_params:
                        removed_idx = j
                        removed_mask = bm
                        break
                        
                if removed_idx is None:
                    continue
                    
                new_mask = current_mask & ~removed_mask
                new_packing = current_packing[:i] + current_packing[i+1:]
                
                # Try to add two blocks that are disjoint from each other and from new_mask
                # We look for blocks that are disjoint from new_mask
                candidates = []
                for j in range(num_blocks_list):
                    if j == removed_idx:
                        continue
                    p, bm, _ = all_blocks[j]
                    if (bm & new_mask) == 0:
                        # Check if p is already in new_packing
                        if p not in new_packing:
                            candidates.append(j)
                            
                if len(candidates) < 2:
                    continue
                    
                # Try all pairs of candidates? That's O(N^2). 
                # To speed up, we can just pick the first two disjoint ones found
                # Or do a small random sample
                found_pair = False
                # Shuffle candidates to avoid bias
                rng.shuffle(candidates)
                
                for a in range(min(50, len(candidates))):
                    if found_pair: break
                    idx_a = candidates[a]
                    mask_a = all_blocks[idx_a][1]
                    for b in range(a + 1, min(50, len(candidates))):
                        idx_b = candidates[b]
                        mask_b = all_blocks[idx_b][1]
                        if (mask_a & mask_b) == 0:
                            # Found a valid pair
                            # Check if they are already in new_packing (should be disjoint from new_mask so likely not, unless they were removed? No, we only removed one)
                            # But we must ensure they are not in new_packing
                            p_a = all_blocks[idx_a][0]
                            p_b = all_blocks[idx_b][0]
                            if p_a not in new_packing and p_b not in new_packing:
                                new_packing.append(p_a)
                                new_packing.append(p_b)
                                new_mask |= mask_a
                                new_mask |= mask_b
                                if len(new_packing) > current_size:
                                    current_packing = new_packing
                                    current_mask = new_mask
                                    current_size = len(current_packing)
                                    improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
                    
            if not improved:
                # If no improvement in a full pass, we can stop or try another strategy
                # For now, break to avoid infinite loops if stuck
                break
                
        return current_packing

    # Main search loop
    rng = random.Random(42)
    
    # Initial best from conflict greedy
    init_packing = greedy_conflict()
    best_num_blocks = len(init_packing)
    best_packing = init_packing
    
    # Try local search on the initial packing
    if time.time() < deadline - 20.0:
        ls_packing = local_search(best_packing, rng, max_iters=100)
        if len(ls_packing) > best_num_blocks:
            best_num_blocks = len(ls_packing)
            best_packing = ls_packing
    
    # Random restarts with different heuristics
    num_trials = 500
    for t in range(num_trials):
        if time.time() > deadline - 20.0:
            break
            
        if t % 3 == 0:
            packing = greedy_random(rng)
        elif t % 3 == 1:
            packing = greedy_rarity(rng)
        else:
            # Hybrid: Random shuffle but sorted by conflict count in case of tie? 
            # Or just another random
            packing = greedy_random(rng)
            
        if len(packing) > best_num_blocks:
            best_num_blocks = len(packing)
            best_packing = packing
            
            # Run local search on the new best
            if time.time() < deadline - 20.0:
                ls_packing = local_search(best_packing, rng, max_iters=50)
                if len(ls_packing) > best_num_blocks:
                    best_num_blocks = len(ls_packing)
                    best_packing = ls_packing
            
            if best_num_blocks >= 12:
                # Try to find 13
                pass
            if best_num_blocks >= 13:
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
    
    # Convert to matrix format (list of lists)
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
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
