# EVOLVE-BLOCK-START
import itertools
import random
import time


def _count_covers(block, blocks, t):
    """Count how many t-subsets of block are covered only by this block."""
    cnt = 0
    for sub in itertools.combinations(block, t):
        only = True
        for b in blocks:
            if b is block:
                continue
            if all(x in b for x in sub):
                only = False
                break
        if only:
            cnt += 1
    return cnt


def _greedy_cover(v, k, t, rng, time_budget):
    """Randomized greedy covering construction."""
    all_points = list(range(v))
    # Represent uncovered t-subsets as a set of frozensets or tuples
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    start_time = time.time()
    
    while uncovered:
        if time.time() - start_time > time_budget:
            break
        
        # Pick a random uncovered t-subset to start the block
        seed_subset = rng.choice(list(uncovered))
        block = list(seed_subset)
        
        # Greedily add points to maximize coverage of uncovered t-subsets
        while len(block) < k:
            best_point = None
            best_gain = -1
            block_set = set(block)
            
            for x in all_points:
                if x in block_set:
                    continue
                # Calculate gain: number of new t-subsets covered
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point is not None:
                block.append(best_point)
                block_set.add(best_point)
            else:
                # No improvement, break
                break
        
        if len(block) < k:
            # Pad with random points if needed
            block_set = set(block)
            remaining = [x for x in all_points if x not in block_set]
            rng.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, time_budget):
    """Remove blocks that don't cover any unique t-subset."""
    start_time = time.time()
    changed = True
    while changed and time.time() - start_time < time_budget:
        changed = False
        for i in range(len(blocks)):
            if time.time() - start_time > time_budget:
                break
            if _count_covers(blocks[i], blocks, t) == 0:
                blocks.pop(i)
                changed = True
                break  # Restart loop since indices changed
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using randomized greedy with redundancy removal.
    """
    total_deadline = 120  # seconds
    start_time = time.time()
    
    # Use multiple random seeds for diversification
    best_blocks = None
    best_count = float('inf')
    
    # Time allocation: 70% for construction, 30% for refinement
    construct_budget = 80
    refine_budget = 30
    
    # Try multiple random constructions
    num_trials = 5
    per_trial_time = construct_budget / num_trials
    
    for trial in range(num_trials):
        elapsed = time.time() - start_time
        if elapsed > construct_budget:
            break
        
        rng = random.Random(42 + trial * 1000)
        
        blocks = _greedy_cover(v, k, t, rng, per_trial_time - (time.time() - start_time))
        
        if not blocks:
            continue
        
        # Remove redundant blocks
        remaining_time = refine_budget - (time.time() - start_time)
        if remaining_time > 5:
            blocks = _remove_redundant_blocks(blocks, v, k, t, remaining_time)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
    
    # Final check: ensure we have a valid covering
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = set(itertools.combinations(range(v), t))
        best_blocks = []
        while uncovered:
            seed_subset = min(uncovered)
            block = list(seed_subset)
            block_set = set(block)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        new_sub = tuple(sorted(rest + (x,)))
                        if new_sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is not None:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break
            if len(block) < k:
                remaining = [x for x in range(v) if x not in block_set]
                block.extend(remaining[:k - len(block)])
            block.sort()
            best_blocks.append(block)
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
    
    return best_blocks
# EVOLVE-BLOCK-END
