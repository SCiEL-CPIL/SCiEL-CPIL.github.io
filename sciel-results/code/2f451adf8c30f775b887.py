"""Randomized greedy covering for C(19,7,4) with time budget management."""
import itertools
import random
import time
import math

# Precompute all 4-subsets of 19 points as tuples
_ALL_T_SUBSETS = list(itertools.combinations(range(19), 4))
_T_SUBSET_SET = set(_ALL_T_SUBSETS)


def _count_uncovered_in_block(block_set, uncovered_set):
    """Count how many uncovered t-subsets are covered by a full k-block."""
    count = 0
    for combo in itertools.combinations(block_set, 4):
        if combo in uncovered_set:
            count += 1
    return count


def _greedy_block_from_seed(seed_points, uncovered_set, v, k, t, rng, use_randomized=False):
    """
    Build a block starting from seed_points, greedily adding points.
    If use_randomized, pick next point with probability proportional to gain (with some noise).
    """
    block = list(seed_points)
    block_set = set(block)
    
    while len(block) < k:
        # Compute gain for each candidate point
        gains = []
        candidates = []
        for x in range(v):
            if x in block_set:
                continue
            # Count how many uncovered t-subsets would be covered by adding x
            gain = 0
            for rest in itertools.combinations(block, t - 1):
                combo = tuple(sorted(rest + (x,)))
                if combo in uncovered_set:
                    gain += 1
            if gain > 0:
                gains.append(gain)
                candidates.append(x)
        
        if not candidates:
            # No point adds coverage; just fill with arbitrary points
            for x in range(v):
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
                    if len(block) == k:
                        break
            break
        
        if use_randomized:
            # Weighted random selection: probability proportional to gain
            # Use exponential weighting to favor high gain but allow exploration
            weights = [g ** 0.5 for g in gains]  # sqrt to soften
            total_w = sum(weights)
            r = rng.random() * total_w
            cumsum = 0
            chosen_idx = len(candidates) - 1
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    chosen_idx = i
                    break
        else:
            # Deterministic: pick max gain
            max_gain = max(gains)
            chosen_idx = gains.index(max_gain)
        
        chosen = candidates[chosen_idx]
        block.append(chosen)
        block_set.add(chosen)
    
    return sorted(block)


def construct(v, k, t):
    """
    Randomized greedy set cover for covering design C(v,k,t).
    Multiple randomized restarts within time budget, keep best result.
    """
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for evaluation overhead
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple randomized trials
    num_trials = 0
    for trial in range(50):  # Max trials, will break on time
        if time.time() - start_time > time_limit - 5:
            break
        
        rng = random.Random(trial * 7919 + 13)
        
        # Initialize uncovered set
        uncovered = set(all_t_subsets)
        blocks = []
        
        while uncovered:
            if time.time() - start_time > time_limit - 2:
                break
            
            # Pick a random uncovered t-subset as seed
            seed_subset = rng.choice(list(uncovered))
            seed_points = list(seed_subset)
            
            # Build a block from this seed
            block = _greedy_block_from_seed(
                seed_points, uncovered, v, k, t, rng, use_randomized=True
            )
            
            # Remove all t-subsets covered by this block
            for combo in itertools.combinations(block, t):
                uncovered.discard(combo)
            
            blocks.append(block)
        
        if not uncovered and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        
        # Also try a more deterministic variant every few trials
        if trial % 5 == 0:
            rng2 = random.Random(trial * 104729 + 37)
            uncovered2 = set(all_t_subsets)
            blocks2 = []
            
            while uncovered2:
                if time.time() - start_time > time_limit - 2:
                    break
                
                # Pick the uncovered subset with fewest existing block overlaps (heuristic)
                # For simplicity, just pick randomly but use deterministic block building
                seed_subset = rng2.choice(list(uncovered2))
                seed_points = list(seed_subset)
                
                block = _greedy_block_from_seed(
                    seed_points, uncovered2, v, k, t, rng2, use_randomized=False
                )
                
                for combo in itertools.combinations(block, t):
                    uncovered2.discard(combo)
                
                blocks2.append(block)
            
            if not uncovered2 and len(blocks2) < best_count:
                best_blocks = blocks2
                best_count = len(blocks2)
        
        num_trials += 1
    
    # If we never completed a full cover (time ran out), fall back to a simple greedy
    if best_blocks is None:
        uncovered = set(all_t_subsets)
        blocks = []
        while uncovered:
            seed = min(uncovered)
            block = list(seed)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for combo in itertools.combinations(block, t):
                uncovered.discard(combo)
        best_blocks = blocks
    
    return best_blocks
