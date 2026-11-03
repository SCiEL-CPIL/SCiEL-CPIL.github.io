"""Randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    used_blocks = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        
        # Greedily add points to maximize coverage of uncovered subsets
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
        if b_tuple not in used_blocks:
            used_blocks.add(b_tuple)
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0

    # Precompute all t-subsets for reference
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    
    def compute_counts(blk_list):
        counts = {}
        for b in blk_list:
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                counts[f] = counts.get(f, 0) + 1
        return counts

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    counts = compute_counts(blocks)
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Simulated Annealing / Local Search Loop
    iteration = 0
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # 1. Try to remove a redundant block
        removed = False
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
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    old_c = counts[f]
                    new_c = old_c - 1
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
            
        # 2. Try to swap a block to enable removal
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
            
            # If no critical subsets, the block is redundant (should have been caught in step 1)
            # But due to race conditions or logic, let's handle it
            if not critical_subsets:
                continue
            
            # Collect points involved in critical subsets
            points_needed = set()
            for fs in critical_subsets:
                points_needed.update(fs)
                
            # If points_needed > k, we cannot cover all critical subsets in one block
            if len(points_needed) > k:
                continue
            
            # Try to form a new block containing points_needed and filling the rest greedily
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedy fill to maximize coverage of low-count subsets
            for x in cands:
                if len(new_block_set) >= k:
                    break
                gain = 0
                for rest in itertools.combinations(new_block_set, t-1):
                    fs = frozenset(list(rest) + [x])
                    c = counts.get(fs, 0)
                    if c == 0:
                        gain += 10
                    elif c == 1:
                        gain += 1
                if gain > 0:
                    new_block_set.add(x)
                elif len(new_block_set) < k:
                    new_block_set.add(x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                
                if new_block == old_block:
                    continue
                
                # Validate the swap
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
                valid = True
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    if counts.get(f, 0) == 0:
                        valid = False
                        break
                    
                if valid:
                    blocks[i] = new_block
                    # Update best if smaller? Size is same, but enables further removals
                else:
                    # Revert
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        counts[f] = old_counts_state.get(f, 0)
                        if counts[f] == 0:
                            del counts[f]
                            
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        if f in counts:
                            counts[f] -= 1
                            if counts[f] == 0:
                                del counts[f]
            
        # 3. Random Restart / Perturbation if stuck
        if iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = compute_counts(blocks)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            # Occasionally try a more aggressive perturbation: replace 2 blocks
            elif iteration % 500 == 0 and len(blocks) > 2:
                # Pick 2 random blocks to replace
                i, j = rng.sample(range(len(blocks)), 2)
                old_blocks = [blocks[i], blocks[j]]
                
                # Collect all critical subsets from these two blocks
                critical_subsets = []
                for b in old_blocks:
                    for fs in itertools.combinations(b, t):
                        f = frozenset(fs)
                        if counts.get(f, 0) == 1:
                            critical_subsets.append(f)
                
                points_needed = set()
                for fs in critical_subsets:
                    points_needed.update(fs)
                
                if len(points_needed) <= 2 * k:
                    # Try to form 2 new blocks covering all critical subsets
                    # For simplicity, just try to form one block that covers as many as possible
                    # and another that covers the rest
                    pass  # This is complex, skip for now

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
