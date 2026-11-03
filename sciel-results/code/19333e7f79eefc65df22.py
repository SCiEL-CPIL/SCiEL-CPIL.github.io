"""Simulated Annealing for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _greedy_cover(v, k, t, rng=None):
    """Fast greedy construction to get a valid starting covering."""
    if rng is None:
        rng = random.Random(42)
    
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
        # Greedily add points to maximize coverage
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
        if b_tuple not in blocks:
            blocks.append(b_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
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
    current_count = len(blocks)
    
    # Precompute all t-subsets for faster checking if needed, 
    # but counts dict is sufficient for coverage check.
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.995
    max_iterations = 500000
    
    iteration = 0
    
    while time.time() - start_time < time_limit and T > T_min:
        iteration += 1
        
        # 1. Try to remove redundant blocks (always accept if valid)
        # Randomly pick a block to check for redundancy
        if current_count > 0:
            i = rng.randrange(current_count)
            b = blocks[i]
            
            # Check if all t-subsets in b have count > 1
            redundant = True
            for fs in itertools.combinations(b, t):
                f = frozenset(fs)
                if counts.get(f, 0) <= 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove block
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                
                blocks.pop(i)
                current_count -= 1
                
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
                
                continue
        
        # 2. Swap move: Replace a block with a new one
        if current_count > 0:
            i = rng.randrange(current_count)
            old_block = blocks[i]
            
            # Find critical t-subsets (count == 1) in old_block
            critical = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            if not critical:
                # No critical subsets, we can try to replace with a better covering block
                # Or just skip for now to focus on reduction
                continue
            
            # Construct new block that covers critical subsets
            # The new block must contain the union of all critical subsets? No, just cover them.
            # Actually, if we remove old_block, critical subsets become uncovered.
            # So new_block must cover all critical subsets.
            # This implies new_block must contain the union of points in critical subsets?
            # No, it just needs to contain each critical subset.
            # Since critical subsets are t-subsets of old_block, their union is a subset of old_block.
            # Let U = union(critical). The new block must cover all f in critical.
            # This is a constraint satisfaction problem.
            
            # Heuristic: Start with union of critical points, then fill rest greedily
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
            
            if len(points_needed) > k:
                # Cannot cover all critical subsets with one block if their union is too large
                # This means we cannot simply swap this block without uncovering something
                # unless we do a multi-block move. For now, skip.
                continue
                
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
            # Fill remaining spots to maximize coverage of currently uncovered or low-count subsets
            while len(new_block_set) < k:
                best_x = None
                best_gain = -1
                # Only consider a few candidates for speed
                sample_cands = cands[:20] if len(cands) > 20 else cands
                for x in sample_cands:
                    gain = 0
                    for rest in itertools.combinations(new_block_set, t - 1):
                        fs = frozenset(list(rest) + [x])
                        c = counts.get(fs, 0)
                        if c == 0:
                            gain += 10
                        elif c == 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is None:
                    break
                new_block_set.add(best_x)
                if best_x in cands:
                    cands.remove(best_x)
                    
            if len(new_block_set) == k:
                new_block = tuple(sorted(new_block_set))
                if new_block != old_block:
                    # Calculate delta in uncovered count
                    delta_uncovered = 0
                    # Remove old block
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        old_c = counts.get(f, 0)
                        if old_c == 1:
                            delta_uncovered += 1
                    # Add new block
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        new_c = counts.get(f, 0)
                        if new_c == 0:
                            delta_uncovered -= 1
                    
                    # Note: This delta calculation is approximate if overlaps exist.
                    # Let's do it properly by simulating.
                    
                    # Simulate remove old
                    sim_counts = dict(counts)
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        sim_counts[f] -= 1
                        if sim_counts[f] == 0:
                            del sim_counts[f]
                    
                    # Simulate add new
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        if f in sim_counts:
                            sim_counts[f] += 1
                        else:
                            sim_counts[f] = 1
                    
                    # Check validity
                    # A move is valid if no t-subset becomes uncovered.
                    # Since we started with a valid cover, and we only remove/add,
                    # we need to ensure that any subset uncovered by removal is covered by addition.
                    # This is equivalent to: for all f, if sim_counts.get(f, 0) == 0, then it was 0 before?
                    # No, if it was 1 before and is 0 now, it's uncovered.
                    
                    valid = True
                    for f in set(list(itertools.combinations(range(v), t))): # This is too slow
                        pass
                    
                    # Faster check:
                    # Any f that was count 1 in old_block and is not covered by new_block is uncovered.
                    # Any f that was count 0 and is covered by new_block is fine.
                    # Any f that was count > 1 is fine.
                    
                    # Let's just check if the new state has any 0 counts for t-subsets that were previously covered?
                    # Actually, we just need to ensure the new state is a valid covering.
                    # Since we are doing SA, we can accept invalid states if we penalize them,
                    # but the problem requires a valid covering at the end.
                    # So we should only accept valid moves or use a penalty.
                    
                    # Let's check validity of the swap.
                    # Uncovered subsets after swap = (Uncovered before) U (Lost from old) - (Gained from new)
                    # Uncovered before is empty (valid cover).
                    # So we need (Lost from old) subset of (Gained from new).
                    
                    lost = set()
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        if counts.get(f, 0) == 1:
                            lost.add(f)
                    
                    gained = set()
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        if counts.get(f, 0) == 0 or counts.get(f, 0) == 1: # If it was 0, it's new gain. If 1, it's redundant gain.
                            # Actually, if count was 0, it becomes 1 (covered).
                            # If count was 1, it becomes 2 (still covered).
                            # We only care if lost subsets are covered by new block.
                            gained.add(f)
                            
                    if lost.issubset(gained):
                        # Valid swap
                        # Apply swap
                        for fs in itertools.combinations(old_block, t):
                            f = frozenset(fs)
                            counts[f] -= 1
                            if counts[f] == 0:
                                del counts[f]
                        for fs in itertools.combinations(new_block, t):
                            f = frozenset(fs)
                            if f in counts:
                                counts[f] += 1
                            else:
                                counts[f] = 1
                        blocks[i] = new_block
                        # current_count unchanged
                        # No improvement in count, but might enable future removals
                    else:
                        # Invalid swap, skip or accept with penalty?
                        # For simplicity, skip invalid swaps in this implementation.
                        # A more advanced SA would use penalties.
                        pass
        
        # 3. Perturbation: Remove a block and greedily repair
        if iteration % 100 == 0:
            if current_count > 1:
                old_blocks = list(blocks)
                old_counts = dict(counts)
                old_current_count = current_count
                
                # Remove a random block
                i = rng.randrange(current_count)
                b = blocks[i]
                for fs in itertools.combinations(b, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                blocks.pop(i)
                current_count -= 1
                
                # Find uncovered
                unc_list = []
                # We don't have a list of all t-subsets readily available for iteration in O(1) per step
                # But we can find uncovered by checking which counts are 0?
                # We don't have a global list of all t-subsets stored.
                # Let's generate them.
                for c in itertools.combinations(range(v), t):
                    f = frozenset(c)
                    if counts.get(f, 0) == 0:
                        unc_list.append(f)
                        
                # Greedily cover uncovered
                while unc_list:
                    seed = rng.choice(unc_list)
                    new_b = list(seed)
                    rng.shuffle(new_b)
                    
                    while len(new_b) < k:
                        best_x = None
                        best_gain = -1
                        cands = [x for x in range(v) if x not in new_b]
                        rng.shuffle(cands)
                        for x in cands:
                            gain = 0
                            for rest in itertools.combinations(new_b, t - 1):
                                fs = frozenset(list(rest) + [x])
                                if counts.get(fs, 0) == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        new_b.append(best_x)
                    
                    new_b.sort()
                    new_b_tuple = tuple(new_b)
                    
                    # Avoid duplicates
                    if new_b_tuple not in blocks:
                        blocks.append(new_b_tuple)
                        for fs in itertools.combinations(new_b_tuple, t):
                            f = frozenset(fs)
                            if f in counts:
                                counts[f] += 1
                            else:
                                counts[f] = 1
                        current_count += 1
                        
                        # Update unc_list
                        # This is expensive. Rebuild?
                        unc_list = []
                        for c in itertools.combinations(range(v), t):
                            f = frozenset(c)
                            if counts.get(f, 0) == 0:
                                unc_list.append(f)
                
                # Check if we improved or stayed same
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = list(blocks)
                elif current_count == best_count:
                    pass
                else:
                    # Accept with probability?
                    # For now, revert if worse to keep it simple and ensure validity
                    # A true SA would accept worse states.
                    # Let's implement a simple acceptance: if worse, revert.
                    blocks = old_blocks
                    counts = old_counts
                    current_count = old_current_count
        
        T *= alpha

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
