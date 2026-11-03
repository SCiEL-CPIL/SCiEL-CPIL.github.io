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


def construct(v, k, t):
    rng = random.Random(42)
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsubs = [frozenset(c) for c in itertools.combinations(range(v), t)]
    total_t = len(all_tsubs)
    
    # Build initial valid covering
    blocks = _greedy_cover(v, k, t, rng)
    
    # Maintain coverage counts: counts[f] = number of blocks covering frozenset f
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
    
    # Track number of uncovered and singly-covered t-subsets
    def count_uncovered_single(counts):
        uncovered = 0
        single = 0
        for f in all_tsubs:
            c = counts.get(f, 0)
            if c == 0:
                uncovered += 1
            elif c == 1:
                single += 1
        return uncovered, single

    uncovered, single = count_uncovered_single(counts)
    
    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.995
    min_temp = 0.01
    
    iteration = 0
    no_improve = 0
    
    while time.time() - start_time < time_limit and temperature > min_temp:
        iteration += 1
        temperature *= cooling_rate
        
        # Strategy 1: Try to remove a redundant block (always accept if valid)
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
                # Remove block i
                blocks.pop(i)
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
                    no_improve = 0
                else:
                    no_improve += 1
                removed = True
                break
        
        if removed:
            continue
            
        # Strategy 2: Block swap - replace one block with another to potentially enable removals
        if len(blocks) > 0:
            i = rng.randrange(len(blocks))
            old_block = blocks[i]
            
            # Find critical subsets (count == 1) in old_block
            critical = []
            for fs in itertools.combinations(old_block, t):
                f = frozenset(fs)
                if counts.get(f, 0) == 1:
                    critical.append(f)
            
            # If no critical subsets, block is redundant (should have been caught in Strategy 1)
            if not critical:
                continue
            
            # Collect points that must be in the new block to cover critical subsets
            points_needed = set()
            for fs in critical:
                points_needed.update(fs)
            
            # If we need more than k points, we can't cover all critical subsets in one block
            if len(points_needed) > k:
                # Try a different strategy: pick a random block and try to swap with a better one
                continue
            
            # Build new block containing points_needed, fill rest greedily
            new_block_set = set(points_needed)
            cands = [x for x in range(v) if x not in new_block_set]
            rng.shuffle(cands)
            
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
                
                # Simulate the swap
                # Save state
                old_state = {}
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    old_state[f] = counts.get(f, 0)
                
                # Remove old_block
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    counts[f] -= 1
                    if counts[f] == 0:
                        del counts[f]
                        
                # Add new_block
                for fs in itertools.combinations(new_block, t):
                    f = frozenset(fs)
                    counts[f] = counts.get(f, 0) + 1
                        
                # Check validity: ensure no subset became uncovered
                valid = True
                new_block_tsubs = set(frozenset(c) for c in itertools.combinations(new_block, t))
                for fs in itertools.combinations(old_block, t):
                    f = frozenset(fs)
                    if f not in counts and f not in new_block_tsubs:
                        valid = False
                        break
                    
                if valid:
                    blocks[i] = new_block
                    no_improve += 1
                else:
                    # Revert
                    for fs in itertools.combinations(old_block, t):
                        f = frozenset(fs)
                        counts[f] = old_state.get(f, 0)
                        if counts[f] == 0:
                            del counts[f]
                    for fs in itertools.combinations(new_block, t):
                        f = frozenset(fs)
                        if f in counts:
                            counts[f] -= 1
                            if counts[f] == 0:
                                del counts[f]
            
        # Strategy 3: Perturbation if stuck for too long
        if no_improve > 500:
            # Try a fresh greedy cover
            new_cover = _greedy_cover(v, k, t, rng)
            if len(new_cover) < len(blocks):
                blocks = new_cover
                counts = compute_counts(blocks)
                uncovered, single = count_uncovered_single(counts)
                if len(blocks) < best_count:
                    best_count = len(blocks)
                    best_blocks = list(blocks)
                    no_improve = 0
            else:
                # Randomly perturb: remove a random block and re-add a new one
                if len(blocks) > 0:
                    i = rng.randrange(len(blocks))
                    old_b = blocks[i]
                    # Remove
                    blocks.pop(i)
                    for fs in itertools.combinations(old_b, t):
                        f = frozenset(fs)
                        counts[f] -= 1
                        if counts[f] == 0:
                            del counts[f]
                    # Add a new random block
                    new_b = tuple(sorted(rng.sample(range(v), k)))
                    blocks.append(new_b)
                    for fs in itertools.combinations(new_b, t):
                        f = frozenset(fs)
                        counts[f] = counts.get(f, 0) + 1
                    no_improve = 0

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
