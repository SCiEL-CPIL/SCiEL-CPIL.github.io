"""Improved program for C(19,7,4) covering design."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def _greedy_cover(v, k, t, rng, deadline):
    """
    Randomized greedy set cover for covering design.
    Returns list of blocks (tuples).
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    subset_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    
    covered = [False] * n_tsubs
    uncovered_count = n_tsubs
    
    blocks = []
    
    # Precompute for each t-subset, the list of points it contains (for quick lookup)
    # Actually, we need: for each point x, which t-subsets contain x
    # But that's large. Instead, we'll compute gain on the fly.
    
    # Precompute combinations of size (t-1) from each possible partial block
    # This is expensive, so we'll do it incrementally
    
    while uncovered_count > 0:
        if time.time() > deadline:
            break
        
        # Find the best block to add
        # Strategy: pick a random uncovered t-subset, then greedily extend it
        # But also consider: among all blocks, pick the one covering most uncovered
        
        # For efficiency with v=19, k=7: C(19,7) = 50388 possible blocks
        # We can't check all each iteration, so use randomized approach
        
        # Pick a random uncovered t-subset as seed
        uncovered_indices = [i for i in range(n_tsubs) if not covered[i]]
        if not uncovered_indices:
            break
        
        seed_idx = rng.choice(uncovered_indices)
        seed = all_tsubs[seed_idx]
        
        # Greedily build a block starting from seed
        block = set(seed)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            
            for x in range(v):
                if x in block:
                    continue
                
                # Count how many currently uncovered t-subsets would be covered
                # by adding x to block
                # A t-subset S is covered if S ⊆ block ∪ {x} and S was uncovered
                # This means S contains x, and S \ {x} ⊆ block
                # So we need: S contains x, S \ {x} ⊆ block, S uncovered
                
                gain = 0
                # S must contain x and (t-1) elements from block
                for combo in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(list(combo) + [x]))
                    if candidate in subset_to_idx:
                        idx = subset_to_idx[candidate]
                        if not covered[idx]:
                            gain += 1
                
                # Add small random perturbation for diversification
                gain += rng.random() * 0.5
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain <= 0.5:  # threshold: no real gain
                # Pick random point
                available = [x for x in range(v) if x not in block]
                if available:
                    best_point = rng.choice(available)
                else:
                    break
            
            block.add(best_point)
        
        if len(block) < k:
            available = [x for x in range(v) if x not in block]
            rng.shuffle(available)
            for x in available:
                if len(block) >= k:
                    break
                block.add(x)
        
        if len(block) < k:
            break
        
        block_tuple = tuple(sorted(block))
        
        # Check if this block adds any new coverage
        new_covered = 0
        for t_subset in itertools.combinations(block_tuple, t):
            idx = subset_to_idx[t_subset]
            if not covered[idx]:
                new_covered += 1
        
        if new_covered == 0:
            continue
        
        blocks.append(block_tuple)
        
        # Update coverage
        for t_subset in itertools.combinations(block_tuple, t):
            idx = subset_to_idx[t_subset]
            if not covered[idx]:
                covered[idx] = True
                uncovered_count -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """
    Try to remove blocks that don't contribute unique coverage.
    Returns reduced list of blocks.
    """
    all_tsubs = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks
    
    # For each block, count how many t-subsets it uniquely covers
    # (i.e., t-subsets covered by this block but not by any other block)
    
    # Build coverage map: for each t-subset, which blocks cover it
    tsub_to_blocks = defaultdict(list)
    for bi, block in enumerate(blocks):
        for t_subset in itertools.combinations(block, t):
            tsub_to_blocks[t_subset].append(bi)
    
    # A block is redundant if every t-subset it covers is also covered by another block
    redundant = [False] * n_blocks
    for bi in range(n_blocks):
        block = blocks[bi]
        for t_subset in itertools.combinations(block, t):
            covering_blocks = tsub_to_blocks[t_subset]
            if len(covering_blocks) == 1:
                # This t-subset is only covered by block bi
                redundant[bi] = False
                break
        else:
            redundant[bi] = True
    
    # Remove redundant blocks
    new_blocks = [blocks[bi] for bi in range(n_blocks) if not redundant[bi]]
    
    return new_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    rng = random.Random(42)
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds
    n_restarts = 10
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        rng_restart = random.Random(42 + restart)
        
        # Run greedy cover
        blocks = _greedy_cover(v, k, t, rng_restart, deadline)
        
        if not blocks:
            continue
        
        # Verify coverage
        all_tsubs = list(itertools.combinations(range(v), t))
        subset_to_idx = {s: i for i, s in enumerate(all_tsubs)}
        covered = [False] * len(all_tsubs)
        
        for block in blocks:
            for t_subset in itertools.combinations(block, t):
                idx = subset_to_idx[t_subset]
                covered[idx] = True
        
        if not all(covered):
            continue
        
        # Try to remove redundant blocks
        if time.time() < deadline - 5:
            blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks[:]
        
        # If we've used too much time, stop
        if time.time() > deadline - 10:
            break
    
    if best_blocks is None:
        # Fallback: run one more greedy without time pressure
        rng_fallback = random.Random(123)
        best_blocks = _greedy_cover(v, k, t, rng_fallback, deadline)
    
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
