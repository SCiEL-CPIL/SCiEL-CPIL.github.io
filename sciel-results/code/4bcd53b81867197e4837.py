"""Randomized local search (simulated annealing) for covering design C(24,7,3)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math


def _greedy_cover(v, k, t, rng):
    """Fast greedy construction to get a valid starting covering."""
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
    
    total_t = len(list(itertools.combinations(range(v), t)))
    uncovered = total_t - len(counts)
    
    def apply_block_change(blk, sign):
        """
        sign = 1 to add, -1 to remove.
        Returns change in uncovered count (delta_uncovered).
        """
        delta_uncovered = 0
        for fs in itertools.combinations(blk, t):
            f = frozenset(fs)
            old_c = counts.get(f, 0)
            new_c = old_c + sign
            if new_c == 0:
                if f in counts:
                    del counts[f]
                if old_c == 1:
                    delta_uncovered += 1
                if old_c == 0:
                    delta_uncovered -= 1
            elif new_c > 0:
                if old_c == 0:
                    delta_uncovered -= 1
                counts[f] = new_c
        return delta_uncovered

    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    
    # Pre-compute all possible blocks? No, C(24,7) is 346k, manageable but memory heavy.
    # We will generate candidates on the fly.
    
    while time.time() - start_time < time_limit:
        
        # 1. Try to remove redundant blocks (Hill Climbing step, always accept if valid)
        removed_any = False
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
                delta_u = apply_block_change(b, -1)
                uncovered += delta_u
                if uncovered == 0:
                    blocks.pop(i)
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    removed_any = True
                    T = max(T_min, T * 0.9) # Cool down on improvement
                    break
                else:
                    apply_block_change(b, 1)
                    uncovered -= delta_u
        
        if removed_any:
            continue
            
        # 2. Simulated Annealing Moves
        if len(blocks) > 0:
            move_type = rng.random()
            
            if move_type < 0.5:
                # Replace a block
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                
                critical = []
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    if counts.get(f, 0) == 1:
                        critical.append(f)
                
                if not critical:
                    # No critical subsets, we can replace with any random block
                    # But we want to minimize count, so replacement is only useful if it enables removals.
                    # For now, skip if no criticals, as it's a neutral move.
                    continue
                    
                points_needed = set()
                for fs in critical:
                    points_needed.update(fs)
                
                if len(points_needed) > k:
                    continue
                
                # Generate a new block containing points_needed
                # Fill rest randomly
                remaining_slots = k - len(points_needed)
                cands = [x for x in range(v) if x not in points_needed]
                rng.shuffle(cands)
                
                # Try a few candidates for the new block
                best_new_block = None
                best_delta_u = 0
                best_gain = -1
                
                for _ in range(10): # Try 10 random completions
                    new_block_set = set(points_needed)
                    for x in rng.sample(cands, min(remaining_slots, len(cands))):
                        new_block_set.add(x)
                    
                    if len(new_block_set) < k:
                        # Fill with any
                        for x in cands:
                            if len(new_block_set) >= k: break
                            new_block_set.add(x)
                            
                    new_block = tuple(sorted(new_block_set))
                    if new_block == old_block:
                        continue
                    if new_block in blocks:
                        continue
                        
                    # Evaluate swap
                    delta_u_remove = apply_block_change(old_block, -1)
                    delta_u_add = apply_block_change(new_block, 1)
                    total_delta_u = delta_u_remove + delta_u_add
                    
                    if total_delta_u < best_delta_u or (total_delta_u == best_delta_u and rng.random() < 0.5):
                        best_delta_u = total_delta_u
                        best_new_block = new_block
                        best_gain = -total_delta_u # Lower uncovered is better
                        
                    # Undo evaluation
                    apply_block_change(new_block, -1)
                    apply_block_change(old_block, 1)
                    
                if best_new_block is not None:
                    # Accept based on SA
                    # Energy change is 0 (count stays same), but uncovered changes.
                    # We treat uncovered > 0 as a penalty.
                    # Let's define Energy = len(blocks) + lambda * uncovered
                    # Here we just check if it improves uncovered or is neutral.
                    if best_delta_u <= 0:
                        # Improve or neutral uncovered
                        apply_block_change(old_block, -1)
                        apply_block_change(best_new_block, 1)
                        uncovered += best_delta_u
                        blocks[i] = best_new_block
                    else:
                        # Worse uncovered, accept with prob exp(-best_delta_u / T)
                        if rng.random() < math.exp(-best_delta_u / T):
                            apply_block_change(old_block, -1)
                            apply_block_change(best_new_block, 1)
                            uncovered += best_delta_u
                            blocks[i] = best_new_block
                            
            else:
                # Add a block (Uphill move, might enable removals)
                # Generate a random block that covers many uncovered sets
                # Or just a random block
                new_block_set = set(rng.sample(range(v), k))
                new_block = tuple(sorted(new_block_set))
                
                if new_block not in blocks:
                    delta_u_add = apply_block_change(new_block, 1)
                    # Energy increases by 1 (count)
                    # Accept with prob exp(-1/T)
                    if rng.random() < math.exp(-1 / T):
                        blocks.append(new_block)
                        uncovered += delta_u_add
                    else:
                        apply_block_change(new_block, -1)
        
        T = max(T_min, T * cooling_rate)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
