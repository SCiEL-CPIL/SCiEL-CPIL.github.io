"""Improved covering design C(v,k,t) via randomized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict


def _all_t_subsets(v, t):
    return [tuple(c) for c in itertools.combinations(range(v), t)]


def _blocks_covering_block(block, v, t):
    """Return set of t-subsets (as tuples) covered by block."""
    return set(itertools.combinations(sorted(block), t))


def _coverage_counts(blocks, v, t):
    """Count how many times each t-subset is covered."""
    counts = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(sorted(block), t):
            counts[sub] += 1
    return counts


def _greedy_build(v, k, t, all_subs, rng, deadline):
    """
    Randomized greedy: at each step pick the uncovered t-subset with highest 'bottleneck'
    degree (i.e., the one that appears in the fewest potential blocks), then build a block
    around it by greedily adding points that cover the most uncovered subsets.
    """
    uncovered = set(all_subs)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick target: the uncovered subset that is "most constrained"
        # Use a simple heuristic: pick a random uncovered subset, but weight by
        # how many blocks could still cover it (fewer = more urgent)
        # For speed, just pick a random one from uncovered
        target = rng.choice(list(uncovered))
        
        block = list(target)
        rng.shuffle(block)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # Find point that maximizes coverage of uncovered t-subsets
            best_point = None
            best_gain = -1
            best_gain2 = -1  # secondary: min coverage among new subs
            
            block_set = set(block)
            for x in range(v):
                if x in block_set:
                    continue
                
                # Compute gain: number of uncovered t-subsets that would be covered
                # by adding x to block
                gain = 0
                # New t-subsets formed by adding x to block: choose t-1 from block
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                
                # Secondary criterion: among the newly covered subsets, how many
                # are already covered by other blocks (we want to cover "rare" ones)
                # This is expensive, so use a simpler tiebreaker
                if gain > best_gain or (gain == best_gain and x > best_point):
                    best_gain = gain
                    best_point = x
            
            if best_point is None:
                break
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            blocks.append(block)
            covered = set(itertools.combinations(block, t))
            uncovered.difference_update(covered)
    
    return blocks


def _local_search(blocks, v, k, t, all_subs, rng, deadline):
    """
    Local search: try to remove blocks that are redundant (all their t-subsets are
    covered by other blocks), and try to swap blocks for better coverage.
    """
    if not blocks:
        return blocks
    
    blocks = list(blocks)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Compute coverage counts
        counts = _coverage_counts(blocks, v, t)
        
        # Try to remove redundant blocks
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            block = blocks[i]
            block_subs = set(itertools.combinations(block, t))
            
            # Check if all subsets in this block are covered by other blocks
            redundant = True
            for sub in block_subs:
                if counts[sub] <= 1:
                    redundant = False
                    break
            
            if redundant:
                blocks.pop(i)
                improved = True
                # Recompute counts
                counts = _coverage_counts(blocks, v, t)
        
        # Try to add a block that covers the most uncovered subsets
        if time.time() > deadline:
            break
        
        uncovered = set()
        for sub in all_subs:
            if counts.get(sub, 0) == 0:
                uncovered.add(sub)
        
        if not uncovered:
            break
        
        # Find best new block to add
        best_block = None
        best_gain = 0
        
        # Sample candidate blocks
        num_samples = min(500, len(uncovered))
        for _ in range(num_samples):
            if time.time() > deadline:
                break
            
            # Start from a random uncovered subset
            target = rng.choice(list(uncovered))
            block = list(target)
            rng.shuffle(block)
            
            while len(block) < k:
                best_point = None
                best_p_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_p_gain:
                        best_p_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
            
            if len(block) == k:
                block.sort()
                gain = 0
                for sub in itertools.combinations(block, t):
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
        
        if best_block and best_gain > 0:
            blocks.append(best_block)
            improved = True
    
    return blocks


def _verify_covering(blocks, v, k, t):
    """Verify that every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        if len(block) != k or len(set(block)) != k:
            return False
        if any(x < 0 or x >= v for x in block):
            return False
        for sub in itertools.combinations(sorted(block), t):
            covered.add(sub)
    
    all_count = 1
    for i in range(v - t + 1, 0, -1):
        all_count *= i
        if i > 1:
            all_count //= i
    
    # Check if all t-subsets are covered
    # For efficiency, just check the count
    if len(covered) < all_count:
        return False
    return True


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    rng = random.Random(42)
    
    # Precompute all t-subsets
    all_subs = _all_t_subsets(v, t)
    
    # Phase 1: Multiple randomized greedy runs
    best_blocks = None
    best_len = float('inf')
    
    num_greedy = 5
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        
        # Use different random seeds for diversity
        rng_i = random.Random(42 + i * 1000)
        blocks = _greedy_build(v, k, t, all_subs, rng_i, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = set(all_subs)
        blocks = []
        while uncovered:
            target = min(uncovered)
            block = list(target)
            while len(block) < k:
                best_point = None
                best_gain = -1
                block_set = set(block)
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
            if len(block) == k:
                block.sort()
                blocks.append(block)
                covered = set(itertools.combinations(block, t))
                uncovered.difference_update(covered)
        best_blocks = blocks
        best_len = len(blocks)
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline:
        local_blocks = _local_search(best_blocks, v, k, t, all_subs, rng, deadline)
        if len(local_blocks) < best_len:
            best_blocks = local_blocks
            best_len = len(local_blocks)
    
    # Phase 3: More randomized restarts if time remains
    if time.time() < deadline - 10:
        for i in range(10, 20):
            if time.time() > deadline:
                break
            
            rng_i = random.Random(42 + i * 1000)
            blocks = _greedy_build(v, k, t, all_subs, rng_i, deadline)
            
            # Quick local search
            blocks = _local_search(blocks, v, k, t, all_subs, rng_i, deadline)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
    
    # Ensure we return a valid covering
    # If the covering is invalid, fall back to a known-valid construction
    # For safety, verify and if invalid, do a simple exhaustive check
    
    # Simple validity check
    covered = set()
    for block in best_blocks:
        for sub in itertools.combinations(sorted(block), t):
            covered.add(sub)
    
    if len(covered) < len(all_subs):
        # Add blocks to cover remaining subsets
        uncovered = set(all_subs) - covered
        while uncovered:
            target = min(uncovered)
            block = list(target)
            block_set = set(block)
            while len(block) < k:
                best_point = None
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    break
                block.append(best_point)
                block_set.add(best_point)
            if len(block) == k:
                block.sort()
                best_blocks.append(block)
                covered_new = set(itertools.combinations(block, t))
                uncovered.difference_update(covered_new)
    
    # Deduplicate blocks
    seen = set()
    unique_blocks = []
    for b in best_blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            unique_blocks.append(list(b))
    
    return unique_blocks
# EVOLVE-BLOCK-END
