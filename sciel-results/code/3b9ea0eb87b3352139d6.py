"""Enhanced portfolio search for covering design C(v,k,t) with advanced local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict


def _compute_cover_count(blocks, v, k, t):
    """Compute how many times each t-subset is covered."""
    cover_count = defaultdict(int)
    for block in blocks:
        for combo in itertools.combinations(block, t):
            cover_count[combo] += 1
    return cover_count


def _greedy_cover(v, k, t, rng, deadline, randomize=False, weight=0.0):
    """
    Greedy covering construction with optional weighting.
    Returns list of blocks (each a sorted tuple of k points).
    """
    all_t_subsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a t-subset to start with
        if randomize and rng.random() < 0.7:
            start = rng.choice(list(uncovered))
        else:
            start = min(uncovered)
        
        block = list(start)
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            
            if not candidates:
                break
            
            # Compute gain for each candidate point
            best_gain = -1
            best_point = None
            best_points = []
            
            for x in candidates:
                g = 0
                for rest in itertools.combinations(block, t - 1):
                    combo = tuple(sorted(rest + (x,)))
                    if combo in uncovered:
                        g += 1
                
                if g > best_gain:
                    best_gain = g
                    best_points = [x]
                elif g == best_gain and g > 0:
                    best_points.append(x)
            
            if best_gain <= 0:
                # No improvement possible, pick randomly
                best_point = rng.choice(candidates)
            elif randomize and len(best_points) > 1:
                # Pick from best points with some randomness
                best_point = rng.choice(best_points)
            else:
                best_point = best_points[0]
            
            block.append(best_point)
        
        if len(block) == k:
            block.sort()
            block_tuple = tuple(block)
            blocks.append(block_tuple)
            
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
    """
    blocks = list(blocks)
    if not blocks:
        return blocks
    
    # Count how many times each t-subset is covered
    cover_count = _compute_cover_count(blocks, v, k, t)
    
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


def _try_replace_block(blocks, v, k, t, idx, deadline, rng):
    """
    Try to replace block at index idx with a better block.
    Returns (new_blocks, success) where success is True if we improved.
    """
    blocks = list(blocks)
    old_block = blocks[idx]
    
    # Compute current cover count
    cover_count = _compute_cover_count(blocks, v, k, t)
    
    # Find t-subsets that are only covered by this block
    unique_to_block = set()
    for combo in itertools.combinations(old_block, t):
        if cover_count[combo] == 1:
            unique_to_block.add(combo)
    
    if not unique_to_block:
        # This block can be removed entirely
        blocks.pop(idx)
        return blocks, True
    
    # Try to find a replacement block that covers all unique_to_block
    # and doesn't add too many new subsets
    best_new_block = None
    best_score = float('inf')
    
    # Generate candidate blocks that contain all of unique_to_block
    # Start with the union of unique_to_block
    core_points = set()
    for combo in unique_to_block:
        core_points.update(combo)
    
    if len(core_points) > k:
        return blocks, False  # Can't fit in a k-block
    
    # Try different completions of the core
    num_attempts = 50
    for _ in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Start with core points
        new_block = list(core_points)
        
        # Add random additional points
        remaining_points = [x for x in range(v) if x not in core_points]
        if remaining_points:
            num_to_add = k - len(new_block)
            if num_to_add > 0:
                additional = rng.sample(remaining_points, min(num_to_add, len(remaining_points)))
                new_block.extend(additional)
        
        if len(new_block) != k:
            continue
        
        new_block.sort()
        new_block_tuple = tuple(new_block)
        
        # Check if this block covers all unique_to_block
        new_covered = set(itertools.combinations(new_block_tuple, t))
        if not unique_to_block.issubset(new_covered):
            continue
        
        # Score: number of new t-subsets covered (lower is better)
        old_covered = set(itertools.combinations(old_block, t))
        new_only = new_covered - old_covered
        score = len(new_only)
        
        if score < best_score:
            best_score = score
            best_new_block = new_block_tuple
    
    if best_new_block is not None:
        blocks[idx] = best_new_block
        return blocks, True
    
    return blocks, False


def _local_search_advanced(blocks, v, k, t, deadline, rng):
    """
    Advanced local search: remove redundant blocks, try replacements, and swaps.
    """
    blocks = list(blocks)
    if not blocks:
        return blocks
    
    # Phase 1: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    # Phase 2: Try to replace blocks
    improved = True
    max_iterations = 200
    iteration = 0
    
    while improved and iteration < max_iterations:
        if time.time() > deadline:
            break
        improved = False
        iteration += 1
        
        # Try each block
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            
            new_blocks, success = _try_replace_block(blocks, v, k, t, i, deadline, rng)
            
            if success and len(new_blocks) < len(blocks):
                blocks = new_blocks
                improved = True
                break
    
    # Phase 3: Final redundant removal
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)
    
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) using enhanced portfolio search.
    """
    start_time = time.time()
    total_time = 150.0  # Wall clock limit
    safety_margin = 15.0
    deadline = start_time + total_time - safety_margin
    
    rng = random.Random(42)
    
    # Portfolio: run multiple greedy constructions with different randomizations
    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Quick greedy runs to get initial solutions
    num_greedy_runs = 8
    greedy_deadline = start_time + (total_time - safety_margin) * 0.4
    
    for run_idx in range(num_greedy_runs):
        if time.time() > greedy_deadline:
            break
        
        # Vary randomization level
        randomize = (run_idx > 0)
        
        blocks = _greedy_cover(v, k, t, rng, greedy_deadline, randomize=randomize)
        
        if not blocks:
            continue
        
        # Check validity
        if not _is_valid_covering(blocks, v, k, t):
            continue
        
        # Quick local search
        local_deadline = min(greedy_deadline, time.time() + 2.0)
        blocks = _local_search_advanced(blocks, v, k, t, local_deadline, rng)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Phase 2: More intensive local search on best solution
    if best_blocks is not None:
        intensive_deadline = start_time + (total_time - safety_margin) * 0.8
        blocks = _local_search_advanced(best_blocks, v, k, t, intensive_deadline, rng)
        
        if _is_valid_covering(blocks, v, k, t) and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = list(blocks)
    
    # Phase 3: Try more diverse solutions with remaining time
    if best_blocks is not None and time.time() < deadline - 10:
        # Run a few more greedy + local search cycles
        for run_idx in range(5):
            if time.time() > deadline - 5:
                break
            
            blocks = _greedy_cover(v, k, t, rng, deadline - 5, randomize=True)
            
            if not blocks:
                continue
            
            if not _is_valid_covering(blocks, v, k, t):
                continue
            
            local_deadline = min(deadline - 2, time.time() + 5.0)
            blocks = _local_search_advanced(blocks, v, k, t, local_deadline, rng)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        blocks = _greedy_cover(v, k, t, rng, deadline, randomize=False)
        best_blocks = blocks
    
    # Convert tuples to lists as required by interface
    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
