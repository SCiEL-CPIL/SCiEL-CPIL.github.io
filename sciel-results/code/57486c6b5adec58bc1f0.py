import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: aim to finish well before 120s
    # The greedy loop is fast, so we can do many restarts
    # Let's allow 110 seconds for the whole process
    
    # Precomputation
    # 1. Map t-subsets to IDs
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    # 2. Precompute point_combo_to_id for fast gain calculation
    # point_combo_to_id[x][combo] = id of t-subset (x, *combo)
    # combo is a tuple of (t-1) points, not containing x
    point_combo_to_id = [None] * v
    for x in range(v):
        point_combo_to_id[x] = {}
        # Generate all (t-1) combinations from points other than x
        # It's simpler to generate all (t-1) combos from 0..v-1 and filter if x in combo
        # Or generate from set(range(v)) - {x}
        other_points = [p for p in range(v) if p != x]
        for combo in itertools.combinations(other_points, t - 1):
            # Form the t-subset
            # combo is sorted, x is just x.
            # We need the sorted tuple to match t_sub_to_id
            ts_tuple = tuple(sorted((x,) + combo))
            point_combo_to_id[x][combo] = t_sub_to_id[ts_tuple]

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    # Estimate time per restart:
    # ~300 blocks * 7 steps * (19 candidates * 35 combos) ~ 1.4M dict lookups
    # Should be < 0.1s per restart in Python.
    # So we can do hundreds of restarts in 110s.
    
    restart_count = 0
    while (time.time() - start_time) < 110:
        restart_count += 1
        uncovered = set(range(num_t_subs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            # Converting set to list is O(N). N decreases.
            # For speed, we can pick a random index from the list of all t_subs 
            # and check if it's uncovered? No, that's inefficient if uncovered is small.
            # list(uncovered) is fine.
            seed_id = random.choice(list(uncovered))
            seed_ts = all_t_subs[seed_id]
            
            block = list(seed_ts)
            
            # Fill block to size k
            while len(block) < k:
                best_x = -1
                best_gain = -1
                candidates = [x for x in range(v) if x not in block]
                
                # Precompute (t-1) subsets of current block for this step
                # This is constant for all candidates in this iteration
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in candidates:
                    gain = 0
                    # Use precomputed map
                    x_map = point_combo_to_id[x]
                    for combo in block_t_minus_1:
                        ts_id = x_map[combo]
                        if ts_id in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_gain > 0:
                        # Random tie-breaking to encourage diversity
                        if random.random() < 0.5:
                            best_x = x
                            
                block.append(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Remove covered t-subsets
            # We can optimize this by only removing those that are in uncovered
            for ts in itertools.combinations(block, t):
                ts_id = t_sub_to_id[ts]
                if ts_id in uncovered:
                    uncovered.remove(ts_id)
                    
        # Post-processing: Remove redundant blocks
        # A block is redundant if all t-subsets it covers are also covered by other blocks.
        # This check is expensive: O(B^2 * C(k,t)).
        # B ~ 300. 300^2 * 56 ~ 5M ops. Might be too slow per restart.
        # Let's skip this for now, or do it only on the best candidate at the end?
        # Or do a simple pass: for each block, check if it's the *only* cover for any t-subset?
        # That's also expensive.
        
        # For now, just count blocks
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # If we find a very good solution (close to lower bound?), we could stop early?
        # Lower bound is ~207. Best known 335.
        # If we find < 335, it's a new record (unlikely but possible).
        # If we find 335, it's as good as known best.
        
    return best_blocks if best_blocks else []
