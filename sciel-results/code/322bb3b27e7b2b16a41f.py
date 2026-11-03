"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import sys


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
    uncovered = [set() for _ in range(v)]
    # uncovered[x] = set of points y such that some t-subset containing x,y is still uncovered
    # Actually, let's track uncovered t-subsets differently for speed.
    
    # Precompute all t-subsets
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered_set = set(all_tsubsets)
    blocks = []
    used_blocks = set()
    
    while uncovered_set:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered_set))
        block_set = set(seed)
        
        # Greedily add points to maximize coverage
        while len(block_set) < k:
            best_x = None
            best_gain = -1
            
            # Try each candidate point
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # Count how many uncovered t-subsets would be covered by adding x
                for rest in itertools.combinations(block_set, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered_set:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain == 0:
                break
            block_set.add(best_x)
            
        block = tuple(sorted(block_set))
        if block not in used_blocks:
            used_blocks.add(block)
            blocks.append(block)
            # Remove covered t-subsets
            for fs in itertools.combinations(block, t):
                uncovered_set.discard(frozenset(fs))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Precompute all t-subsets for quick lookup
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubsets)
    
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
    
    # Maintain uncovered_count = number of t-subsets with count 0
    def count_uncovered(counts):
        unc = 0
        for fs in all_tsubsets:
            if counts.get(fs, 0) == 0:
                unc += 1
        return unc
    
    uncovered_count = count_uncovered(counts)
    
    # Simulated Annealing / Local Search Loop
    temp = 1.0
    cooling = 0.99995
    iteration = 0
    
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
                    if new_c == 0:
                        uncovered_count += 1
                        del counts[f]
                    else:
                        counts[f] = new_c
                
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                removed = True
                break
        
        if removed:
            continue
            
        # 2. Try to swap a block to enable further removals
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Identify critical subsets (count == 1) in old_block
            critical_subsets = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical_subsets.append(f)
            
            if len(critical_subsets) == 0:
                # No critical subsets, try to remove (should have been caught in step 1)
                # But maybe we missed it due to randomness, so try again
                continue
            
            # Collect points needed to cover all critical subsets
            points_needed = set()
            for fs in critical_subsets:
                points_needed.update(fs)
                
            # If we can't fit all needed points in one block, skip
            if len(points_needed) > k:
                continue
                
            # Try to form a new block containing points_needed
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Greedy fill to maximize coverage of uncovered or single-covered subsets
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
                    
                # Validate the swap: simulate removal and addition
                # Check if any subset would become uncovered
                valid = True
                # Subsets in old_block but not in new_block might lose their last cover
                new_block_tsubsets = set(frozenset(c) for c in itertools.combinations(new_block, t))
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    if counts.get(f, 0) == 1 and f not in new_block_tsubsets:
                        valid = False
                        break
                
                if valid:
                    # Accept swap
                    # Update counts: remove old_block, add new_block
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        old_c = counts[f]
                        new_c = old_c - 1
                        if new_c == 0:
                            del counts[f]
                        else:
                            counts[f] = new_c
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        counts[f] = counts.get(f, 0) + 1
                    blocks[i] = new_block
                else:
                    # Try a different approach: random perturbation
                    # Replace with a random block that covers critical subsets
                    if len(points_needed) <= k:
                        # Try multiple random fills
                        for attempt in range(5):
                            new_block_set = set(points_needed)
                            cands = [x for x in range(v) if x not in new_block_set]
                            rng.shuffle(cands)
                            for x in cands:
                                if len(new_block_set) >= k:
                                    break
                                new_block_set.add(x)
                            if len(new_block_set) == k:
                                new_block = tuple(sorted(new_block_set))
                                if new_block == old_block:
                                    continue
                                new_block_tsubsets = set(frozenset(c) for c in itertools.combinations(new_block, t))
                                valid = True
                                for fs in itertools.combinations(old_block, t):
                                    f = frozenset(fs)
                                    if counts.get(f, 0) == 1 and f not in new_block_tsubsets:
                                        valid = False
                                        break
                                if valid:
                                    for fs in itertools.combinations(old_block, t):
                                        f = frozenset(fs)
                                        old_c = counts[f]
                                        new_c = old_c - 1
                                        if new_c == 0:
                                            del counts[f]
                                        else:
                                            counts[f] = new_c
                                    for fs in itertools.combinations(new_block, t):
                                        f = frozenset(fs)
                                        counts[f] = counts.get(f, 0) + 1
                                    blocks[i] = new_block
                                    break
    
        # 3. Random restart if stuck for too long
        if iteration % 200 == 0:
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = compute_counts(blocks)
                uncovered_count = count_uncovered(counts)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            else:
                # Try to improve current blocks with a fresh greedy pass on uncovered subsets
                # This is complex, so just continue
                pass
    
        # Cool down
        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
