"""Covering design C(v, k, t) – optimized greedy with deadline guard."""
import itertools
import time
import random

_DEADLINE_S = 112.0  # Safety margin from 120s evaluator limit
_START_TIME = time.time()


def _time_left():
    return max(0.0, _DEADLINE_S - (time.time() - _START_TIME))


def _greedy_block(uncovered, v, k, t, seed, rng):
    """Build a k-subset containing seed, maximizing coverage of uncovered t-subsets."""
    block = list(seed)
    block_set = set(block)
    # Precompute: for each point x, which (t-1)-subsets of current block combined with x
    # are in uncovered. We'll update incrementally.
    
    while len(block) < k and _time_left() > 0.3:
        best_pt = None
        best_gain = -1
        best_tiebreak = -1
        for x in range(v):
            if x in block_set:
                continue
            # Compute gain: number of uncovered t-subsets that would be covered
            # by adding x to block
            gain = 0
            for combo in itertools.combinations(block, t - 1):
                sub = tuple(sorted(combo + (x,)))
                if sub in uncovered:
                    gain += 1
            # Tiebreak: prefer points that appear in more uncovered subsets overall
            if gain > best_gain or (gain == best_gain and best_pt is not None):
                # Compute tiebreak score: how many uncovered t-subsets contain x
                # This is expensive, so only compute when needed
                if best_pt is not None:
                    tiebreak = sum(1 for s in uncovered if x in s)
                else:
                    tiebreak = 0
                if gain > best_gain or (gain == best_gain and tiebreak > best_tiebreak):
                    best_gain = gain
                    best_pt = x
                    best_tiebreak = tiebreak
        
        if best_pt is None:
            candidates = [x for x in range(v) if x not in block_set]
            if not candidates:
                break
            best_pt = rng.choice(candidates)
        
        block.append(best_pt)
        block_set.add(best_pt)
    
    # Fill if needed
    if len(block) < k:
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block_set:
                block.append(x)
                block_set.add(x)
    return sorted(block)


def _greedy_block_fast(uncovered, v, k, t, seed, rng):
    """Faster greedy: precompute gain scores for all points at once."""
    block = list(seed)
    block_set = set(block)
    
    while len(block) < k and _time_left() > 0.3:
        # Compute gain for each candidate point
        gains = {}
        for x in range(v):
            if x in block_set:
                continue
            gain = 0
            for combo in itertools.combinations(block, t - 1):
                sub = tuple(sorted(combo + (x,)))
                if sub in uncovered:
                    gain += 1
            gains[x] = gain
        
        if not gains:
            break
        
        # Select best point
        max_gain = max(gains.values())
        candidates = [x for x, g in gains.items() if g == max_gain]
        if len(candidates) == 1:
            best_pt = candidates[0]
        else:
            # Tiebreak: prefer point in more uncovered subsets
            best_pt = max(candidates, key=lambda x: sum(1 for s in uncovered if x in s))
        
        block.append(best_pt)
        block_set.add(best_pt)
    
    # Fill if needed
    if len(block) < k:
        for x in range(v):
            if len(block) >= k:
                break
            if x not in block_set:
                block.append(x)
                block_set.add(x)
    return sorted(block)


def construct(v, k, t):
    random.seed(12345)
    all_tsubs = list(itertools.combinations(range(v), t))
    best_blocks = None
    best_len = float('inf')
    
    # Multiple randomized restarts with different strategies
    max_restarts = 40
    for r in range(max_restarts):
        if _time_left() < 2.0:
            break
        
        rng = random.Random(12345 + r * 7919)
        uncovered = set(all_tsubs)
        blocks = []
        block_set = set()
        
        while uncovered and _time_left() > 0.5:
            # Pick target: prioritize "hard" uncovered subsets
            # A subset is "hard" if it has fewer available points that could extend it
            # to a k-block without using too many already-covered points
            
            # Strategy 1: Random with slight bias toward rare points
            # Strategy 2: Prioritize subsets with points that appear in few existing blocks
            # Strategy 3: Pure random
            
            if len(uncovered) > 500:
                # Sample for efficiency
                sample_size = min(200, len(uncovered))
                sample = rng.sample(list(uncovered), sample_size)
            else:
                sample = list(uncovered)
            
            # Score each candidate target by "difficulty"
            # Higher score = harder to cover = should prioritize
            scored_targets = []
            for target in sample:
                # Count how many of the target's points appear in existing blocks
                # Fewer appearances = harder to cover
                point_freq = {}
                for block in blocks:
                    for p in block:
                        point_freq[p] = point_freq.get(p, 0) + 1
                difficulty = sum(1 for p in target if p not in point_freq)
                # Add some randomness
                scored_targets.append((difficulty + rng.random(), target))
            
            scored_targets.sort(reverse=True)
            # Pick from top candidates with some randomness
            top_n = min(10, len(scored_targets))
            target = scored_targets[rng.randint(0, top_n - 1)][1]
            
            block = _greedy_block_fast(uncovered, v, k, t, seed=target, rng=rng)
            bkey = tuple(block)
            if bkey not in block_set:
                blocks.append(block)
                block_set.add(bkey)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        
        if not uncovered:
            if len(blocks) < best_len:
                best_blocks = blocks
                best_len = len(blocks)
                if best_len <= 151:  # Known best
                    break
    
    # Fallback if no valid covering found in time
    if best_blocks is None:
        uncovered = set(all_tsubs)
        blocks = []
        block_set = set()
        rng = random.Random(99999)
        while uncovered and _time_left() > 0.1:
            target = min(uncovered)
            block = _greedy_block_fast(uncovered, v, k, t, seed=target, rng=rng)
            bkey = tuple(block)
            if bkey not in block_set:
                blocks.append(block)
                block_set.add(bkey)
                for sub in itertools.combinations(block, t):
                    uncovered.discard(sub)
        best_blocks = blocks
    
    return best_blocks
