"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage of uncovered triples
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            
        block.sort()
        b_tuple = tuple(block)
        if b_tuple not in block_set:
            block_set.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Precompute all t-subsets for quick lookup if needed, but we use counts
    # counts[fs] = number of blocks covering fs
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] = counts.get(f, 0) + 1
        return counts

    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Helper to check if a list of blocks is valid using counts
    # We don't need to recompute everything, just check min count >= 1
    # But for search, we update counts incrementally.
    
    def is_valid_counts(counts):
        # Check if all t-subsets are covered
        # We need to know the total number of t-subsets
        total_t = len(list(itertools.combinations(range(v), t)))
        # Instead of checking all, we can just ensure no count is 0 for any existing subset?
        # No, we must ensure ALL subsets are covered.
        # It's faster to maintain a count of 'uncovered' subsets.
        pass

    # Let's maintain `uncovered_count` = number of t-subsets with count 0.
    # And `single_count` = number of t-subsets with count 1.
    
    def init_stats(counts):
        uncovered = 0
        single = 0
        total_t = len(list(itertools.combinations(range(v), t)))
        # We need to check all possible t-subsets? 
        # The counts dict only has keys for subsets that appear in blocks.
        # Subsets not in counts have count 0.
        # So, we can't easily get `uncovered` without iterating all C(v,t) subsets.
        # C(24,3) = 2024. This is small enough to iterate every time? 
        # 2024 iterations is fast. Let's do that for safety.
        
        for fs in itertools.combinations(range(v), t):
            f = frozenset(fs)
            c = counts.get(f, 0)
            if c == 0:
                uncovered += 1
            elif c == 1:
                single += 1
        return uncovered, single

    uncovered, single = init_stats(counts)
    
    def try_remove_block(blk_idx):
        """
        Try to remove block at blk_idx.
        Returns (new_blocks, new_counts, new_uncovered, new_single) if valid, else None.
        """
        b = blocks[blk_idx]
        # Check if removal is valid: all t-subsets in b must have count > 1
        for fs in itertools.combinations(b, t):
            f = frozenset(fs)
            if counts.get(f, 0) <= 1:
                return None
        
        # It is valid. Perform removal.
        new_blocks = blocks[:blk_idx] + blocks[blk_idx+1:]
        
        # Update counts
        new_counts = {}
        # Copying dict is O(N). N is small (blocks * C(k,t)).
        # Or we can update in place and revert? 
        # Let's update in place for speed, then revert if we don't accept.
        
        # Since we are in a trial, let's just simulate.
        # Actually, let's just update the global `counts` and `uncovered`/`single` if we accept.
        # For now, return the new state.
        
        # To avoid deep copying, we will modify global variables if accepted.
        # For the trial, we just need to know it's valid.
        return new_blocks

    # Simulated Annealing / Local Search Loop
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        removed = False
        # Randomly check blocks for redundancy
        indices = list(range(len(blocks)))
        rng.shuffle(indices)
        for i in indices:
            b = blocks[i]
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove it
                blocks.pop(i)
                # Update counts
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    old_c = counts[f]
                    new_c = old_c - 1
                    if new_c == 0:
                        uncovered += 1
                    elif new_c == 1:
                        single += 1
                    elif old_c == 1:
                        single -= 1
                    counts[f] = new_c
                    if new_c == 0:
                        del counts[f]
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                removed = True
                break
        
        if removed:
            continue
            
        # 2. Try to swap a block to enable removal or just find a better configuration
        # Strategy: Pick a block that has "low" utility (covers many subsets that are already well-covered)
        # Or pick a block that is "critical" (covers unique subsets) and try to replace it with a block that covers those unique subsets + others.
        
        if len(blocks) > 0:
            # Pick a random block to potentially replace
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Identify subsets in old_block that are critical (count 1)
            critical_subsets = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical_subsets.append(f)
            
            # If no critical subsets, we can just try to remove it? 
            # We already checked removal in step 1. If it failed, it means it has critical subsets.
            # So critical_subsets is not empty.
            
            # We need to construct a new block that covers all critical_subsets.
            # If len(critical_subsets) > 1, the new block must contain the union of these subsets?
            # No, it must contain each subset.
            # Since k=7, t=3.
            # If we have two critical subsets {a,b,c} and {a,b,d}, the new block must contain a,b,c,d.
            # Let's collect the points involved in critical subsets.
            points_needed = set()
            for fs in critical_subsets:
                points_needed.update(fs)
                
            # If points_needed > k, we cannot cover all critical subsets in one block.
            # In this case, we cannot simply replace this block with one block.
            # We might need to do a 2-block swap or revert.
            # For simplicity, if we can't form a valid replacement, we skip.
            if len(points_needed) <= k:
                # Try to form a new block containing points_needed and filling the rest greedily
                new_block_set = set(points_needed)
                # Fill remaining slots to maximize coverage of other subsets
                # Candidates: all points not in new_block_set
                cands = [x for x in range(v) if x not in new_block_set]
                rng.shuffle(cands)
                
                # Greedy fill
                for x in cands:
                    if len(new_block_set) >= k:
                        break
                    # Calculate gain: how many *uncovered* or *single-covered* subsets does adding x cover?
                    # We want to cover subsets that are currently 0 or 1.
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t-1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10 # High priority
                        elif c == 1:
                            gain += 1
                    if gain > 0:
                        new_block_set.add(x)
                    elif len(new_block_set) < k:
                        # If no gain, just add random to fill
                        new_block_set.add(x)
                        
                if len(new_block_set) == k:
                    new_block = tuple(sorted(new_block_set))
                    
                    # Check if new_block is same as old_block
                    if new_block == old_block:
                        continue
                        
                    # Validate the swap:
                    # 1. Remove old_block from counts
                    # 2. Add new_block to counts
                    # 3. Check if all counts >= 1
                    # 4. Revert if invalid
                    
                    # Simulate removal
                    temp_counts = {}
                    # This is getting slow with copying.
                    # Let's do in-place update and revert.
                    
                    # Store old state
                    old_counts_state = {}
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        old_counts_state[f] = counts.get(f, 0)
                    
                    # Apply removal
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        counts[f] -= 1
                        if counts[f] == 0:
                            del counts[f]
                            
                    # Apply addition
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        counts[f] = counts.get(f, 0) + 1
                        
                    # Check validity: Are there any 0s?
                    # We need to check if any subset became 0.
                    # Subsets that were 1 and are in old_block but not new_block become 0.
                    valid = True
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        if f not in counts: # count is 0
                            # Check if new_block covers it
                            if f not in [frozenset(c) for c in itertools.combinations(new_block, t)]:
                                valid = False
                                break
                    # Also check if any NEW subset added by new_block was 0 before? No, adding increases count.
                    # The only risk is subsets that lose their last cover.
                    
                    if valid:
                        # Accept swap
                        blocks[i] = new_block
                        # Update best if smaller? Size is same.
                        # But we might enable further removals.
                        # Recalculate uncovered/single for next iteration?
                        # We can just let the next loop handle it.
                        # But we need accurate `uncovered` and `single` for next step?
                        # The `try_remove` logic uses `counts` directly, so it's fine.
                        # The `init_stats` is only called at start.
                        # We should update `uncovered` and `single` variables?
                        # Actually, `try_remove` doesn't use `uncovered`/`single` variables, it checks `counts`.
                        # So we don't strictly need to update the counters unless we use them for logic.
                        # We don't use them for logic in `try_remove`.
                        pass
                    else:
                        # Revert
                        # Restore old_block
                        for fs in itertools.combinations(old_block, t):
                            f = frozenset(fs)
                            counts[f] = old_counts_state.get(f, 0)
                            if counts[f] == 0:
                                del counts[f]
                            
                        # Remove new_block
                        for fs in itertools.combinations(new_block, t):
                            f = frozenset(fs)
                            if f in counts:
                                counts[f] -= 1
                                if counts[f] == 0:
                                    del counts[f]
            
        # 3. Random Restart / Perturbation if stuck
        if iteration % 100 == 0:
            # Try a new greedy cover
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = compute_counts(blocks)
                uncovered, single = init_stats(counts)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
