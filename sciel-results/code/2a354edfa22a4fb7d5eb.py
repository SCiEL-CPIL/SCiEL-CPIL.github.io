import itertools
import time
import random
import sys
import math

_global_deadline = None

def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = _global_deadline if _global_deadline else float('inf')
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            # Compute gain for each candidate point
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets are completed by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # Remove all t-subsets covered by this block
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)
    return blocks

def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search: iteratively remove redundant
    blocks (those whose removal doesn't leave any t-subset uncovered)."""
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage: for each t-subset, count how many blocks cover it
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if removing block bi leaves any t-subset with count 1
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block bi
                removed_block = blocks.pop(bi)
                # Decrement counts for all t-subsets in removed block
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
    return blocks

def _precompute_blocks(v, k, t):
    """Precompute all possible blocks and their t-subsets."""
    all_blocks = list(itertools.combinations(range(v), k))
    block_subsets = []
    for b in all_blocks:
        subs = set(tuple(sorted(s)) for s in itertools.combinations(b, t))
        block_subsets.append(subs)
    return all_blocks, block_subsets

def _sa_search(blocks, v, k, t, deadline, all_blocks, block_subsets):
    """Simulated Annealing search to minimize block count."""
    if not blocks:
        return blocks
    
    # Map block tuple to index
    block_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    
    # Convert current blocks to indices
    current_indices = set()
    for b in blocks:
        current_indices.add(block_to_idx[tuple(b)])
        
    # Initialize coverage counts
    coverage_count = {}
    for idx in current_indices:
        for ts in block_subsets[idx]:
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
            
    T = 1.0
    best_indices = set(current_indices)
    best_size = len(current_indices)
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        # Pick a random block to remove
        if not current_indices:
            break
        idx_out = rng.choice(list(current_indices))
        
        # Tentatively remove
        current_indices.remove(idx_out)
        uncovered = set()
        for ts in block_subsets[idx_out]:
            coverage_count[ts] -= 1
            if coverage_count[ts] == 0:
                uncovered.add(ts)
                del coverage_count[ts] # Clean up
            elif coverage_count[ts] < 0:
                # Should not happen
                coverage_count[ts] = 0
                
        if not uncovered:
            # Removal was safe (redundant block)
            # Accept
            # No change to coverage_count needed (already decremented)
            pass
        else:
            # Find best block to add
            best_idx_in = -1
            best_overlap = -1
            for idx_in in range(len(all_blocks)):
                if idx_in in current_indices:
                    continue
                # Calculate overlap with uncovered
                # Optimization: if idx_in is idx_out, skip (already removed, but it's not in current_indices)
                overlap = len(block_subsets[idx_in] & uncovered)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx_in = idx_in
                    
            if best_idx_in == -1 or best_overlap < len(uncovered):
                # Cannot repair? Or no block covers all uncovered.
                # If best_overlap < len(uncovered), we can't fully repair with one block.
                # For simplicity, only accept moves that fully repair or improve size.
                # If we can't fully repair, revert.
                # Revert
                current_indices.add(idx_out)
                for ts in block_subsets[idx_out]:
                    coverage_count[ts] = coverage_count.get(ts, 0) + 1
                continue
            
            # Add best_idx_in
            current_indices.add(best_idx_in)
            for ts in block_subsets[best_idx_in]:
                coverage_count[ts] = coverage_count.get(ts, 0) + 1
                if ts in uncovered:
                    # It is now covered, but we don't remove from uncovered set here,
                    # we just rely on coverage_count.
                    pass
        
        # Check if current state is better
        current_size = len(current_indices)
        if current_size < best_size:
            best_size = current_size
            best_indices = set(current_indices)
            
        # Annealing schedule
        T *= 0.999
        if T < 0.01:
            T = 0.01

    # Convert best_indices back to blocks
    return [list(all_blocks[i]) for i in best_indices]

def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 100  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    
    # Phase 2: Local search (reduction)
    if time.time() < _global_deadline - 10:
        blocks = _local_search(blocks, v, k, t, _global_deadline - 10)
        
    # Phase 3: Precompute and SA search
    if time.time() < _global_deadline - 5:
        all_blocks, block_subsets = _precompute_blocks(v, k, t)
        blocks = _sa_search(blocks, v, k, t, _global_deadline - 2, all_blocks, block_subsets)

    return blocks
