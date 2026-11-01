"""Covering design C(24,7,3): fast greedy with randomized restarts and local search."""
import itertools
import time
import random
from collections import defaultdict


def _build_greedy(v, k, t, rng):
    """Build a greedy covering with randomized tie-breaking and start selection."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        # Pick a random uncovered t-subset to start the block
        start = rng.choice(list(uncovered))
        block = list(start)
        block_set = set(block)
        
        while len(block) < k:
            # For each candidate point, count how many uncovered t-subsets it would cover
            best_x = -1
            best_gain = -1
            best_ties = []
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_ties = [x]
                elif gain == best_gain:
                    best_ties.append(x)
            
            # Break ties randomly
            chosen = rng.choice(best_ties) if best_ties else best_x
            block.append(chosen)
            block_set.add(chosen)
        
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        for comb in itertools.combinations(block, t):
            uncovered.discard(tuple(comb))
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """Try to remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks
    
    # Count coverage for each t-subset
    coverage = defaultdict(int)
    for block in blocks:
        for comb in itertools.combinations(block, t):
            coverage[tuple(comb)] += 1
    
    # Try removing each block if it doesn't leave any uncovered t-subset
    new_blocks = list(blocks)
    changed = True
    while changed:
        changed = False
        for i in range(len(new_blocks)):
            # Temporarily remove block i
            new_blocks.pop(i)
            # Check if all t-subsets are still covered
            coverage2 = defaultdict(int)
            for block in new_blocks:
                for comb in itertools.combinations(block, t):
                    coverage2[tuple(comb)] += 1
            
            # Check if any t-subset is uncovered
            all_covered = True
            for comb in itertools.combinations(range(v), t):
                if coverage2[tuple(comb)] == 0:
                    all_covered = False
                    break
            
            if all_covered:
                changed = True
            else:
                new_blocks.insert(i, blocks[i])
    
    return new_blocks


def _local_search(blocks, v, k, t, rng, time_limit=10):
    """Local search: try replacing blocks with better ones."""
    if not blocks:
        return blocks
    
    uncovered = set(itertools.combinations(range(v), t))
    for block in blocks:
        for comb in itertools.combinations(block, t):
            uncovered.discard(tuple(comb))
    
    if not uncovered:
        return blocks
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < time_limit and iterations < 50:
        iterations += 1
        # Try to replace a random block with a new greedy block that covers more uncovered
        idx = rng.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[idx]
        
        # Build a new greedy block starting from an uncovered triple
        if not uncovered:
            break
        start = rng.choice(list(uncovered))
        new_block = list(start)
        block_set = set(new_block)
        
        while len(new_block) < k:
            best_x = -1
            best_gain = -1
            best_ties = []
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(new_block, t - 1):
                    triple = tuple(sorted(rest + (x,)))
                    if triple in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
                    best_ties = [x]
                elif gain == best_gain:
                    best_ties.append(x)
            
            chosen = rng.choice(best_ties) if best_ties else best_x
            new_block.append(chosen)
            block_set.add(chosen)
        
        new_block.sort()
        
        # Check if replacing old_block with new_block reduces uncovered set
        old_covered = set(tuple(c) for c in itertools.combinations(old_block, t))
        new_covered = set(tuple(c) for c in itertools.combinations(new_block, t))
        
        # Uncovered after removing old_block
        temp_uncovered = set(uncovered) | old_covered
        # Then remove new_block's coverage
        temp_uncovered -= new_covered
        
        if len(temp_uncovered) < len(uncovered):
            best_blocks[idx] = new_block
            uncovered = temp_uncovered
            best_len = len(best_blocks)
    
    # Try to remove redundant blocks
    best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
    
    return best_blocks


def construct(v, k, t):
    """Main entry point: try multiple greedy builds with local search."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    best_blocks = None
    best_len = float('inf')
    
    rng = random.Random(42)
    
    # Do several greedy builds with different random seeds
    max_builds = 5
    for build_idx in range(max_builds):
        if time.time() > deadline - 20:
            break
        
        blocks = _build_greedy(v, k, t, rng)
        
        # Local search to improve
        blocks = _local_search(blocks, v, k, t, rng, time_limit=15)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've reached a good number, stop early
        if best_len <= 76:
            break
    
    if best_blocks is None:
        # Fallback: single deterministic greedy
        best_blocks = _build_greedy(v, k, t, random.Random(0))
    
    return best_blocks
# EVOLVE-BLOCK-END
