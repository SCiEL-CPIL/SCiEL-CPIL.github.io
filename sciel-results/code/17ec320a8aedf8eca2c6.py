"""Optimized randomized local search for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def _fast_greedy_cover(v, k, t, rng):
    """Fast greedy construction."""
    all_tsubsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    uncovered = set(all_tsubsets)
    blocks = []
    block_set = set()
    
    while uncovered:
        # Pick a random uncovered t-subset as seed
        seed = rng.choice(list(uncovered))
        block = set(seed)
        
        # Greedily add points to maximize coverage of uncovered subsets
        while len(block) < k:
            best_x = None
            best_gain = -1
            cands = [x for x in range(v) if x not in block]
            rng.shuffle(cands)
            
            for x in cands:
                gain = 0
                # Check subsets formed by x and t-1 elements from block
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(list(rest) + [x])
                    if fs in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.add(best_x)
            
        block_tuple = tuple(sorted(block))
        if block_tuple not in block_set:
            blocks.append(block_tuple)
            block_set.add(block_tuple)
            for c in itertools.combinations(block, t):
                uncovered.discard(frozenset(c))
                
    return blocks


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer

    # 1. Initial Cover
    blocks = _fast_greedy_cover(v, k, t, rng)
    block_set = set(blocks)
    
    # Precompute all t-subsets for the block to speed up updates
    # C(7,3) = 35. This is small.
    
    def get_block_subsets(b):
        return [frozenset(c) for c in itertools.combinations(b, t)]

    # Initialize counts
    counts = Counter()
    for b in blocks:
        for fs in get_block_subsets(b):
            counts[fs] += 1
            
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Helper to check if a block is redundant
    def is_redundant(b):
        for fs in get_block_subsets(b):
            if counts[fs] <= 1:
                return False
        return True

    # Helper to remove a block from state
    def remove_block(b):
        for fs in get_block_subsets(b):
            counts[fs] -= 1
            if counts[fs] == 0:
                del counts[fs]
        blocks.remove(b)
        block_set.remove(b)

    # Helper to add a block to state
    def add_block(b):
        for fs in get_block_subsets(b):
            counts[fs] += 1
        blocks.append(b)
        block_set.add(b)

    # Local Search Loop
    no_improve_count = 0
    max_no_improve = 5000 # Restart if stuck for too long
    
    while time.time() - start_time < time_limit:
        if no_improve_count > max_no_improve:
            # Restart with new greedy
            blocks = _fast_greedy_cover(v, k, t, rng)
            block_set = set(blocks)
            counts = Counter()
            for b in blocks:
                for fs in get_block_subsets(b):
                    counts[fs] += 1
            no_improve_count = 0
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = list(blocks)
            continue

        # Phase 1: Remove redundant blocks
        removed_any = False
        # Iterate backwards to allow safe removal
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            if is_redundant(b):
                remove_block(b)
                removed_any = True
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                no_improve_count = 0 # Reset on improvement
                break # Restart scan after removal
        
        if removed_any:
            continue
            
        # Phase 2: Swap to escape local minima
        # Pick a random block
        if not blocks:
            break
            
        i = rng.randrange(len(blocks))
        old_block = blocks[i]
        
        # Find critical subsets (count 1)
        critical = []
        for fs in get_block_subsets(old_block):
            if counts[fs] == 1:
                critical.append(fs)
                
        if not critical:
            # Should have been removed in Phase 1
            continue
            
        # Determine points needed to cover all critical subsets
        points_needed = set()
        for fs in critical:
            points_needed.update(fs)
            
        # If we can't fit all critical points in one block, we can't do a simple 1-for-1 swap
        if len(points_needed) > k:
            no_improve_count += 1
            continue
            
        # Construct new block
        new_block_set = set(points_needed)
        # Fill remaining slots
        # We want to pick points that maximize "redundancy potential"
        # i.e., points that form subsets with high counts (already covered)
        # This makes the new block more likely to be redundant later.
        
        remaining_slots = k - len(new_block_set)
        candidates = [x for x in range(v) if x not in new_block_set]
        rng.shuffle(candidates)
        
        # Sort candidates by potential "badness" (we want high counts, so low "uncovered" potential)
        # Actually, let's just greedily pick points that create subsets with count > 1
        # Score = sum of counts of subsets formed by x and existing new_block_set
        # We prefer high scores (more redundancy)
        
        scored_candidates = []
        for x in candidates:
            score = 0
            # Subsets formed by x and t-1 elements from new_block_set
            for rest in itertools.combinations(new_block_set, t - 1):
                fs = frozenset(list(rest) + [x])
                # If count is > 1, it's "safe" (redundant)
                # If count is 1, it's "critical"
                # If count is 0, it's "uncovered" (bad, but we are swapping so we assume validity is maintained by covering criticals)
                # Wait, we MUST cover the criticals from old_block.
                # The new block covers points_needed.
                # Any subset in new_block that is NOT in points_needed might be 0?
                # No, we are ADDING the new block. So counts increase.
                # The danger is subsets in old_block that are NOT in new_block.
                # We ensured new_block covers all criticals of old_block.
                # So all subsets of old_block are either:
                # 1. In new_block (count stays same or increases)
                # 2. Not in new_block (count decreases by 1)
                #   If count was 1, it becomes 0 -> INVALID.
                #   But we covered all criticals (count 1) in points_needed.
                #   So any subset of old_block with count 1 is in points_needed, hence in new_block.
                #   So no subset of old_block becomes 0.
                #   Subsets of old_block with count > 1 become count-1 (still >= 1).
                #   So validity is guaranteed if new_block covers points_needed.
                
                # So we are free to pick the rest of the points to optimize for future removals.
                # We want the new block to be "redundant".
                # A block is redundant if all its subsets have count > 1 (in the final state).
                # In the final state, counts for subsets in new_block are:
                #   old_count (from other blocks) + 1 (from new block)
                #   EXCEPT for subsets that were in old_block and new_block?
                #   No, we removed old_block first.
                #   So counts are:
                #   For fs in new_block:
                #     count_new[fs] = count_current[fs] - (1 if fs in old_block else 0) + 1
                #   We want count_new[fs] > 1 for all fs in new_block.
                #   count_current[fs] - delta + 1 > 1
                #   count_current[fs] - delta > 0
                #   If fs in old_block: count_current[fs] - 1 > 0 => count_current[fs] > 1
                #   If fs not in old_block: count_current[fs] > 0 => count_current[fs] >= 1
                #   
                #   So we want to pick points such that:
                #   1. For fs in new_block \ old_block: count_current[fs] >= 1
                #   2. For fs in new_block \cap old_block: count_current[fs] > 1
                #   
                #   Let's score candidates based on this.
                
                if x in old_block:
                    # Subsets formed by x and rest from new_block_set
                    # If rest is also in old_block, we need count > 1
                    # If rest is not in old_block, we need count >= 1
                    for rest in itertools.combinations(new_block_set, t - 1):
                        fs = frozenset(list(rest) + [x])
                        c = counts[fs]
                        if fs in [frozenset(c2) for c2 in itertools.combinations(old_block, t)]:
                            # This is a bit slow. Let's precompute old_subsets
                            pass
                        # Simplified: Just prefer high counts
                        score += c
                else:
                    for rest in itertools.combinations(new_block_set, t - 1):
                        fs = frozenset(list(rest) + [x])
                        score += counts[fs]
                        
            scored_candidates.append((score, x))
            
        # Pick the best candidate
        if scored_candidates:
            scored_candidates.sort(reverse=True)
            # Add top candidates
            for score, x in scored_candidates:
                if len(new_block_set) >= k:
                    break
                new_block_set.add(x)
                
        if len(new_block_set) == k:
            new_block = tuple(sorted(new_block_set))
            
            # Check if new_block is distinct from existing blocks
            if new_block not in block_set:
                # Perform swap
                # 1. Remove old_block
                for fs in get_block_subsets(old_block):
                    counts[fs] -= 1
                    if counts[fs] == 0:
                        del counts[fs]
                blocks[i] = new_block
                block_set.remove(old_block)
                
                # 2. Add new_block
                for fs in get_block_subsets(new_block):
                    counts[fs] += 1
                block_set.add(new_block)
                
                no_improve_count += 1 # Swap doesn't reduce count, but changes structure
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
