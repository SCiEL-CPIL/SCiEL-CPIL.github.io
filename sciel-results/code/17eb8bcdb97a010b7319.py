"""Covering design C(19,8,5): randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def _greedy_pass(v, k, t, rng, time_limit=120.0, start_time=None):
    """One greedy pass: repeatedly pick a t-subset to cover, build a block greedily."""
    if start_time is None:
        start_time = time.time()
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Pick a random uncovered t-subset (or the first for determinism in early passes)
        # Using random choice to escape local optima
        target = rng.choice(list(uncovered))
        block = list(target)
        
        while len(block) < k:
            if time.time() - start_time > time_limit - 5.0:
                break
            
            # Find the point that covers the most uncovered t-subsets
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain <= 0:
                # No point adds coverage; just pick a random one
                available = [x for x in range(v) if x not in block]
                if available:
                    best_x = rng.choice(available)
                else:
                    break
            
            block.append(best_x)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Remove all t-subsets of this block from uncovered
            for tsub in itertools.combinations(block, t):
                uncovered.discard(tsub)
    
    return blocks, uncovered


def _remove_redundant_blocks(v, k, t, blocks, time_limit=120.0, start_time=None):
    """Remove blocks that don't cover any t-subset uniquely."""
    if start_time is None:
        start_time = time.time()
    
    if not blocks:
        return blocks
    
    # Count how many blocks cover each t-subset
    tsub_counts = Counter()
    for block in blocks:
        for tsub in itertools.combinations(block, t):
            tsub_counts[tsub] += 1
    
    # Keep only blocks that cover at least one t-subset with count 1
    kept = []
    for block in blocks:
        if time.time() - start_time > time_limit - 5.0:
            kept.append(block)
            continue
        for tsub in itertools.combinations(block, t):
            if tsub_counts[tsub] == 1:
                kept.append(block)
                break
    
    return kept


def construct(v, k, t):
    start_time = time.time()
    time_limit = 120.0
    
    # Try multiple randomized restarts, keep the best result
    best_blocks = None
    best_len = float('inf')
    
    # Use different seeds for diversity
    num_restarts = 10
    
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10.0:
            break
        
        rng = random.Random(42 + restart)
        
        # Greedy pass
        blocks, remaining = _greedy_pass(v, k, t, rng, time_limit, start_time)
        
        if not blocks:
            continue
        
        # If we didn't finish (remaining uncovered), try to complete or skip
        if remaining:
            # Try to finish the remaining with a simple greedy
            elapsed = time.time() - start_time
            if elapsed < time_limit - 8.0:
                for tsub in remaining:
                    if time.time() - start_time > time_limit - 5.0:
                        break
                    block = list(tsub)
                    while len(block) < k:
                        if time.time() - start_time > time_limit - 5.0:
                            break
                        best_x = None
                        best_gain = -1
                        for x in range(v):
                            if x in block:
                                continue
                            gain = 0
                            for rest in itertools.combinations(block, t - 1):
                                new_tsub = tuple(sorted(rest + (x,)))
                                if new_tsub in remaining:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            available = [x for x in range(v) if x not in block]
                            if available:
                                best_x = rng.choice(available)
                            else:
                                break
                        block.append(best_x)
                    if len(block) == k:
                        block.sort()
                        blocks.append(tuple(block))
                        for ts in itertools.combinations(block, t):
                            remaining.discard(ts)
        
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(v, k, t, blocks, time_limit, start_time)
        
        # Verify coverage
        all_tsubs = set(itertools.combinations(range(v), t))
        covered = set()
        for block in blocks:
            for tsub in itertools.combinations(block, t):
                covered.add(tsub)
        
        if covered >= all_tsubs and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        best_blocks, _ = _greedy_pass(v, k, t, rng, time_limit, start_time)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
