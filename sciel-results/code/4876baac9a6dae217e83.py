"""Covering design C(v,k,t): randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _greedy_pass(v, k, t, rng=None):
    """One greedy pass. Returns list of blocks."""
    if rng is None:
        rng = random.Random()
    
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        start_sub = rng.choice(list(uncovered))
        block = list(start_sub)
        
        # Add points greedily
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        # Remove covered t-subsets
        for comb in itertools.combinations(block, t):
            uncovered.discard(comb)
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """Try to remove redundant blocks from a valid covering."""
    if not blocks:
        return blocks
    
    # Create a set of all t-subsets covered by each block
    block_tsubs = []
    for b in blocks:
        tsubs = set(itertools.combinations(b, t))
        block_tsubs.append(tsubs)
    
    # Union of all t-subsets
    all_tsubs = set()
    for tsubs in block_tsubs:
        all_tsubs.update(tsubs)
    
    # Try to remove each block if its t-subsets are all covered by others
    kept = []
    for i, b in enumerate(blocks):
        # Check if all t-subsets in block i are covered by other blocks
        covered = set()
        for j, tsubs in enumerate(block_tsubs):
            if j != i:
                covered.update(tsubs)
        if block_tsubs[i].issubset(covered):
            continue  # This block is redundant
        kept.append(b)
    
    return kept


def construct(v, k, t):
    """Find a covering design using randomized greedy with redundancy removal."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin: 110s out of 150s limit
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy passes
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        blocks = _greedy_pass(v, k, t, rng)
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've found a good solution, we can stop early
        if best_len <= 340:  # Close to known best of 335
            break
        
        attempt += 1
        # Increase randomness diversity
        rng = random.Random(42 + attempt)
    
    if best_blocks is None:
        # Fallback: single deterministic pass
        best_blocks = _greedy_pass(v, k, t)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
    
    return best_blocks
# EVOLVE-BLOCK-END
