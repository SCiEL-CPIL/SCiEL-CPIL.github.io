import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the exact tensor decomposition for multiplying a 3x5 matrix by a 5x7 matrix.
    Uses a backtracking search to find a maximum packing of disjoint 2x2x2 Strassen blocks.
    Each Strassen block reduces the rank by 1 compared to schoolbook (7 vs 8 terms).
    Total rank = 7 * (num_blocks) + (105 - 8 * num_blocks) = 105 - num_blocks.
    """
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 5, 7

    U_rows = n * m  # 15
    V_rows = m * p  # 35
    W_rows = p * n  # 21

    # Generate all possible 2x2x2 blocks
    all_blocks = []
    block_indices_list = []
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
                                        indices.add((i, j, k))
                            all_blocks.append((i1, i2, j1, j2, k1, k2))
                            block_indices_list.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Precompute conflicts
    # conflict[i] is a bitmask or set of indices j such that block i and block j conflict
    # Using sets of indices for blocks
    conflicts = [set() for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if block_indices_list[i] & block_indices_list[j]:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Heuristic: Sort blocks by number of conflicts (most constrained first)
    # This often helps in backtracking
    order = sorted(range(num_blocks), key=lambda x: len(conflicts[x]), reverse=True)
    
    best_num_blocks = 0
    best_packing = []
    
    # Backtracking search
    # To avoid deep recursion issues or timeout, we limit depth or use iterative deepening?
    # Max possible is 13.
    
    # We can also use a greedy initialization to set a good lower bound
    def greedy_init():
        available = set(range(num_blocks))
        packing = []
        while available:
            # Pick block with min conflicts among available
            best_b = -1
            min_conf = float('inf')
            for b in available:
                c = len(conflicts[b] & available)
                if c < min_conf:
                    min_conf = c
                    best_b = b
            packing.append(best_b)
            available.remove(best_b)
            # Remove all conflicting blocks
            to_remove = conflicts[best_b] & available
            for b in to_remove:
                available.remove(b)
        return packing

    init_packing = greedy_init()
    best_num_blocks = len(init_packing)
    best_packing = init_packing
    
    # Try to improve with local search / swaps?
    # Or use randomized greedy with more trials?
    
    # Let's try a simple randomized greedy with many trials first, as it's robust.
    # The previous code did 10000. Let's do 50000 if time permits.
    
    # Actually, let's try a proper backtracking with pruning.
    # State: current_packing (list of indices), used_indices (frozenset)
    # This might be slow.
    
    # Alternative: Use the greedy result and try to "fix" it or swap elements.
    
    # Let's stick to a very aggressive randomized greedy with a better selection strategy.
    # Strategy: At each step, pick a block from the top K candidates (by lowest conflict count) randomly.
    
    best_num_blocks = 0
    best_packing = []
    
    num_trials = 20000
    for t in range(num_trials):
        if time.time() > deadline - 10.0:
            break
            
        available = set(range(num_blocks))
        current_packing = []
        
        while available:
            # Calculate conflict counts for all available blocks
            # This is O(N^2) per step, which is slow.
            # Optimization: Maintain conflict counts dynamically?
            # For N=630, N^2 = 400k. 100 steps * 400k = 40M ops per trial. Too slow for 20k trials.
            
            # So we need a faster greedy.
            # Precomputed conflicts allow fast lookup.
            # But calculating "min conflicts" requires iterating all available.
            
            # Let's just use random shuffle as before, but maybe fewer trials and faster?
            # The original 10k trials probably took a few seconds.
            
            pass

    # Let's revert to the original strategy but optimize the inner loop.
    # The original used `random.shuffle(all_blocks)` and then picked first available.
    # This is O(N^2) per trial due to set intersection checks?
    # `block_indices & used_indices` is fast if sets are small.
    
    # Let's try to find if 8 blocks are possible.
    # If we find 8, rank is 97. Score 105/97 = 1.082.
    # If we find 7, rank is 98. Score 105/98 = 1.071.
    
    # We will use a hybrid: Randomized Greedy + Local Search (1-swap).
    
    def run_greedy():
        available = set(range(num_blocks))
        packing = []
        # Sort available by random key to break ties? Or just shuffle indices.
        # We can't shuffle the set easily.
        # Let's pick a random block from available, or the one with min conflicts?
        # Min conflicts is better.
        
        # To speed up, we can pre-sort blocks by degree and pick the first available one in that order?
        # No, that's deterministic.
        
        # Let's pick the block with the minimum number of conflicts with other *available* blocks.
        # To make it fast, we can approximate or just use a random block.
        # The original code used random shuffle of the list of blocks, then iterated.
        # This is equivalent to random permutation.
        
        # Let's stick to the original random permutation approach as it's simple and likely sufficient if we increase trials.
        # But we need to check if 7 is the max.
        
        # Let's try to verify the max packing size with a small backtracking search limited to depth 8.
        
        pass

    # Backtracking for max packing up to depth 8 or 9
    # We use the conflict graph.
    
    max_depth = 9
    found_8 = False
    found_9 = False
    best_packing = []
    best_count = 0
    
    # Use the greedy init as a starting point for backtracking?
    # Or just run backtracking from scratch with good ordering.
    
    # Order blocks by degree (descending) to prune earlier?
    # Actually, for independent set, ordering by degree (descending) is standard.
    
    # We will run a depth-first search.
    # To avoid explosion, we use a time limit.
    
    search_start = time.time()
    
    def backtrack(current_packing, available_set):
        nonlocal best_count, best_packing, found_8, found_9
        
        if len(current_packing) > best_count:
            best_count = len(current_packing)
            best_packing = list(current_packing)
            if best_count >= 8: found_8 = True
            if best_count >= 9: found_9 = True
            
        if time.time() > deadline - 5.0:
            return
        
        # Pruning: if current + available <= best, return
        if len(current_packing) + len(available_set) <= best_count:
            return
            
        # Pick a block to branch on
        # Heuristic: Pick block with highest degree in the induced subgraph of available_set?
        # Or just pick the first one in a fixed order?
        
        # Fixed order: sorted by global degree descending.
        # We need to iterate through the fixed order and find the first one in available_set.
        
        # To make this faster, we can maintain a list of available blocks in the fixed order.
        # But available_set changes.
        
        # Simple approach: Iterate through `order` (sorted by degree desc).
        # Find the first block in `available_set`.
        
        candidate = -1
        for b in order:
            if b in available_set:
                candidate = b
                break
                
        if candidate == -1:
            return
            
        # Branch 1: Include candidate
        new_packing = current_packing + [candidate]
        new_available = available_set - conflicts[candidate]
        new_available.remove(candidate)
        
        backtrack(new_packing, new_available)
        
        # Branch 2: Exclude candidate
        # We must remove candidate from available
        new_available2 = available_set - {candidate}
        backtrack(current_packing, new_available2)
        
    # Initial available set
    init_available = set(range(num_blocks))
    
    # To help the search, we can seed it with the greedy solution?
    # Backtracking from scratch might be slow if it doesn't find a good solution quickly.
    # Let's use the greedy solution as a lower bound.
    
    # Run a few random greedy trials to get a good lower bound
    for _ in range(100):
        if time.time() > deadline - 50.0: break
        avail = set(range(num_blocks))
        pack = []
        while avail:
            # Pick random
            b = random.choice(list(avail))
            pack.append(b)
            avail.remove(b)
            avail -= conflicts[b]
        if len(pack) > best_count:
            best_count = len(pack)
            best_packing = pack

    # Now run backtracking to see if we can exceed best_count
    # To prevent explosion, we can limit the search.
    # If best_count is 7, we want to find 8.
    
    # We will run the backtracking search.
    # It might take a while. We give it 60 seconds.
    
    backtrack([], init_available)
    
    # If we found a packing, use it.
    # Otherwise, use the best greedy one.
    
    final_packing = best_packing
    num_blocks_found = len(final_packing)
    
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
    for block_idx in final_packing:
        i1, i2, j1, j2, k1, k2 = all_blocks[block_idx]
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
