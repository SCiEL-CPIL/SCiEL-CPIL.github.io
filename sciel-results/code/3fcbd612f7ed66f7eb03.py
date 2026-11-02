"""Portfolio search for covering design C(v,k,t) with local block removal."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline, randomize=False):
    """
    Greedy covering construction.
    If randomize is True, add some randomness to point selection.
    Returns list of blocks (each a sorted tuple of k points).
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        # Pick a random uncovered t-subset if randomizing, else the first
        if randomize and rng.random() < 0.5:
            start = rng.choice(sorted(uncovered))
        else:
            start = min(uncovered)
        
        block = list(start)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            # Compute gain for each candidate point
            block_set = set(block)
            best_gain = -1
            best_point = None
            candidates = [x for x in range(v) if x not in block_set]
            
            if randomize:
                # Evaluate a random subset of candidates to save time, but still pick best among them
                # Actually for correctness we need good choices; let's evaluate all but add noise
                gains = []
                for x in candidates:
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            g += 1
                    # Add small random noise for diversity
                    gains.append((g + rng.random() * 0.1, x))
                gains.sort(reverse=True)
                if gains:
                    # Pick from top few with some randomness
                    top_n = min(5, len(gains))
                    best_gain, best_point = gains[rng.randint(0, top_n - 1)]
            else:
                for x in candidates:
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        if tuple(sorted(rest + (x,))) in uncovered:
                            g += 1
                    if g > best_gain:
                        best_gain = g
                        best_point = x
                if best_point is None:
                    best_point = candidates[0] if candidates else None
            
            if best_point is None:
                break
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(tuple(block))
            # Remove covered t-subsets
            for combo in itertools.combinations(block, t):
                uncovered.discard(combo)
    
    return blocks


def _is_valid_covering(blocks, v, k, t):
    """Check if blocks cover all t-subsets."""
    covered = set()
    for block in blocks:
        for combo in itertools.combinations(block, t):
            covered.add(combo)
    total = len(list(itertools.combinations(range(v), t)))
    return len(covered) == total


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """
    Try to remove blocks that are redundant (all their t-subsets are covered by other blocks).
    Greedy: repeatedly try to remove a block if it doesn't leave any t-subset uncovered.
    """
    blocks = list(blocks)
    if not blocks:
        return blocks
    
    # Count how many times each t-subset is covered
    cover_count = defaultdict(int)
    for block in blocks:
        for combo in itertools.combinations(block, t):
            cover_count[combo] += 1
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try to remove each block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if removing block i leaves all its t-subsets still covered
            removable = True
            for combo in itertools.combinations(blocks[i], t):
                if cover_count[combo] <= 1:
                    removable = False
                    break
            if removable:
                # Remove block i
                for combo in itertools.combinations(blocks[i], t):
                    cover_count[combo] -= 1
                blocks.pop(i)
                improved = True
                break  # Restart scan from beginning
    
    return blocks


def _local_search(blocks, v, k, t, deadline, rng):
    """
    Local search: try to replace a block with a better one, or remove redundant blocks.
    """
    blocks = list(blocks)
    if not blocks:
        return blocks
    
    # First, remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    # Try to improve by replacing blocks
    # For each block, try to find a replacement that covers more uncovered or same
    # This is complex; for now, just do redundant removal
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using portfolio search.
    """
    start_time = time.time()
    total_time = 150.0  # Wall clock limit
    safety_margin = 10.0
    deadline = start_time + total_time - safety_margin
    
    rng = random.Random(42)
    
    # Portfolio: run multiple greedy constructions with different randomizations
    best_blocks = None
    best_len = float('inf')
    
    num_runs = 20  # Number of portfolio runs
    run_time_budget = (total_time - safety_margin) / num_runs
    
    for run_idx in range(num_runs):
        run_deadline = start_time + (run_idx + 1) * run_time_budget
        if time.time() > run_deadline - 1.0:
            break
        
        # Vary randomization level
        randomize = (run_idx > 0)  # First run is deterministic
        
        blocks = _greedy_cover(v, k, t, rng, run_deadline, randomize=randomize)
        
        if not blocks:
            continue
        
        # Check validity
        if not _is_valid_covering(blocks, v, k, t):
            continue
        
        # Local search to reduce block count
        local_deadline = min(run_deadline, time.time() + run_time_budget * 0.3)
        blocks = _local_search(blocks, v, k, t, local_deadline, rng)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks = _greedy_cover(v, k, t, rng, deadline, randomize=False)
        best_blocks = blocks
    
    # Convert tuples to lists as required by interface
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
