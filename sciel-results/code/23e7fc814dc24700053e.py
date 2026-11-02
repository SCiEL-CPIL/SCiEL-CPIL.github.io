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
        seed = rng.choice(list(uncovered))
        block = list(seed)
        rng.shuffle(block)
        
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


def _compute_counts(blocks, t):
    counts = {}
    for b in blocks:
        for fs in itertools.combinations(b, t):
            f = frozenset(fs)
            counts[f] = counts.get(f, 0) + 1
    return counts


def _count_single(counts, v, t):
    """Count t-subsets with exactly one cover."""
    single = 0
    for fs in itertools.combinations(range(v), t):
        f = frozenset(fs)
        if counts.get(f, 0) == 1:
            single += 1
    return single


def _block_tsubsets(block, t):
    """Yield frozensets of t-subsets of a block."""
    for fs in itertools.combinations(block, t):
        yield frozenset(fs)


def _swap_block(blocks, counts, i, new_block, v, t, rng):
    """
    Try to swap blocks[i] with new_block.
    Returns True if swap was accepted, False otherwise.
    Modifies blocks and counts in place if accepted.
    """
    old_block = blocks[i]
    if old_block == new_block:
        return False
    
    # Compute the set of t-subsets in old_block and new_block
    old_subsets = list(_block_tsubsets(old_block, t))
    new_subsets = list(_block_tsubsets(new_block, t))
    
    # Check validity: every subset in old_block must still be covered after swap
    # i.e., for each s in old_subsets, either count[s] > 1 or s is in new_subsets
    new_subset_set = set(new_subsets)
    for s in old_subsets:
        c = counts.get(s, 0)
        if c <= 1 and s not in new_subset_set:
            return False
    
    # Save old state for potential revert
    old_state = {}
    for s in old_subsets:
        old_state[s] = counts.get(s, 0)
    
    # Apply removal of old_block
    for s in old_subsets:
        counts[s] -= 1
        if counts[s] == 0:
            del counts[s]
    
    # Apply addition of new_block
    for s in new_subsets:
        counts[s] = counts.get(s, 0) + 1
    
    # Accept the swap
    blocks[i] = new_block
    return True


def _try_remove_block(blocks, counts, i, v, t):
    """
    Try to remove blocks[i]. Returns True if removed, False otherwise.
    Modifies blocks and counts in place if removed.
    """
    b = blocks[i]
    for fs in itertools.combinations(b, t):
        f = frozenset(fs)
        if counts.get(f, 0) <= 1:
            return False
    
    # Valid removal
    blocks.pop(i)
    for fs in itertools.combinations(b, t):
        f = frozenset(fs)
        counts[f] -= 1
        if counts[f] == 0:
            del counts[f]
    return True


def _generate_replacement_block(critical_points, blocks, counts, v, k, t, rng):
    """
    Generate a replacement block that contains critical_points and optimizes coverage.
    critical_points: set of points that must be in the new block.
    """
    if len(critical_points) > k:
        return None
    
    new_block_set = set(critical_points)
    
    # Greedy fill remaining slots
    cands = [x for x in range(v) if x not in new_block_set]
    rng.shuffle(cands)
    
    for x in cands:
        if len(new_block_set) >= k:
            break
        gain = 0
        for rest in itertools.combinations(new_block_set, t - 1):
            fs = frozenset(list(rest) + [x])
            c = counts.get(fs, 0)
            if c == 0:
                gain += 100  # High priority for uncovered
            elif c == 1:
                gain += 10   # Medium priority for singly covered
            else:
                gain += 1    # Low priority for well-covered
        new_block_set.add(x)
    
    if len(new_block_set) < k:
        # Should not happen since v >= k
        return None
    
    return tuple(sorted(new_block_set))


def construct(v, k, t):
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0  # leave 5s buffer
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    counts = _compute_counts(blocks, t)
    
    best_blocks = list(blocks)
    best_count = len(blocks)
    
    # Simulated annealing parameters
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    
    stuck_counter = 0
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant blocks (repeat until no more)
        removed_any = True
        while removed_any:
            removed_any = False
            indices = list(range(len(blocks)))
            rng.shuffle(indices)
            for i in indices:
                if _try_remove_block(blocks, counts, i, v, t):
                    removed_any = True
                    if len(blocks) < best_count:
                        best_count = len(blocks)
                        best_blocks = list(blocks)
                    break  # Restart shuffle after removal
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = list(blocks)
        
        # 2. Try swaps with simulated annealing
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Identify critical subsets (count == 1) in old_block
            critical_subsets = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical_subsets.append(f)
            
            # Points that must be in replacement block
            points_needed = set()
            for fs in critical_subsets:
                points_needed.update(fs)
            
            if len(points_needed) <= k:
                new_block = _generate_replacement_block(points_needed, blocks, counts, v, k, t, rng)
                if new_block is not None and new_block != old_block:
                    # Simulate the swap to evaluate
                    old_subsets = list(_block_tsubsets(old_block, t))
                    new_subsets = list(_block_tsubsets(new_block, t))
                    
                    # Compute delta in "single-covered" count
                    old_single = sum(1 for s in old_subsets if counts.get(s, 0) == 1)
                    
                    # After swap, check which subsets become single or uncovered
                    # Subsets in old_block but not new_block: lose one cover
                    # Subsets in new_block but not old_block: gain one cover
                    
                    old_set = set(old_subsets)
                    new_set = set(new_subsets)
                    
                    # Subsets losing a cover (in old, not in new)
                    losing = old_set - new_set
                    # Subsets gaining a cover (in new, not in old)
                    gaining = new_set - old_set
                    
                    # Check if any subset becomes uncovered
                    valid = True
                    for s in losing:
                        if counts.get(s, 0) <= 1:
                            valid = False
                            break
                    
                    if valid:
                        # Evaluate swap quality: prefer swaps that reduce single-covered count
                        # or create more redundancy
                        delta_single = 0
                        for s in losing:
                            c = counts.get(s, 0)
                            if c == 1:
                                delta_single += 1  # Becomes uncovered (bad, but valid means c>1)
                            elif c == 2:
                                delta_single += 1  # Becomes single
                        for s in gaining:
                            c = counts.get(s, 0)
                            if c == 0:
                                delta_single -= 1  # Was uncovered, now covered
                            elif c == 1:
                                delta_single -= 1  # Was single, now double-covered
                        
                        # Accept if delta_single < 0 (improvement) or with probability exp(-delta_single/T)
                        if delta_single < 0:
                            accept = True
                        else:
                            accept = rng.random() < math.exp(-delta_single / T)
                        
                        if accept:
                            if _swap_block(blocks, counts, i, new_block, v, t, rng):
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                        else:
                            stuck_counter += 1
                    else:
                        stuck_counter += 1
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
        
        # 3. Perturbation if stuck
        if stuck_counter > 200:
            # Try a random perturbation: swap 2-3 random blocks
            n_perturb = rng.randint(2, 3)
            for _ in range(n_perturb):
                i = rng.randrange(len(blocks))
                old_block = blocks[i]
                critical_subsets = []
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    if counts.get(f, 0) == 1:
                        critical_subsets.append(f)
                
                points_needed = set()
                for fs in critical_subsets:
                    points_needed.update(fs)
                
                if len(points_needed) <= k:
                    new_block = _generate_replacement_block(points_needed, blocks, counts, v, k, t, rng)
                    if new_block is not None and new_block != old_block:
                        _swap_block(blocks, counts, i, new_block, v, t, rng)
            
            # Try removals again
            removed_any = True
            while removed_any:
                removed_any = False
                indices = list(range(len(blocks)))
                rng.shuffle(indices)
                for i in indices:
                    if _try_remove_block(blocks, counts, i, v, t):
                        removed_any = True
                        if len(blocks) < best_count:
                            best_count = len(blocks)
                            best_blocks = list(blocks)
                        break
            
            stuck_counter = 0
        
        # 4. Occasional restart with new greedy cover
        if stuck_counter > 1000:
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = _compute_counts(blocks, t)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
            stuck_counter = 0
        
        # Cool down
        T = max(T_min, T * cooling_rate)

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
