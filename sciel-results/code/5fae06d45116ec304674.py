"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

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
    # Use a dict mapping t-subset tuple -> count
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


def _block_swap_search(blocks, v, k, t, deadline, rng=None):
    """Local search: try to replace pairs of blocks with a single block that
    covers the same t-subsets. Also try single-block removals."""
    if rng is None:
        rng = random.Random()
    blocks = [list(b) for b in blocks]
    
    # Precompute coverage counts
    coverage_count = {}
    for block in blocks:
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            coverage_count[ts] = coverage_count.get(ts, 0) + 1
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # First pass: try removing redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                return blocks
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1:
                    redundant = False
                    break
            if redundant:
                removed_block = blocks.pop(bi)
                for ts in itertools.combinations(removed_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                improved = True
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # Second pass: try swapping two blocks for one
        # This is expensive, so only do it occasionally
        if rng.random() < 0.3:
            bi = rng.randrange(len(blocks))
            bj = rng.randrange(len(blocks))
            if bi == bj:
                continue
            
            # Check if union of t-subsets from blocks bi and bj can be covered by one block
            # This is a relaxation - we just try to find a single block that covers
            # many of the t-subsets from both blocks
            b_i = set(blocks[bi])
            b_j = set(blocks[bi])
            b_j = set(blocks[bj])
            
            # Try to find a k-subset that covers t-subsets from both blocks
            # Start with the union, try to reduce to k elements
            candidate = sorted(b_i | b_j)
            if len(candidate) <= k:
                candidate_block = candidate
            else:
                # Try greedy reduction
                candidate_block = list(rng.sample(candidate, k))
            
            # Check if this candidate block covers all t-subsets that were ONLY
            # covered by blocks bi or bj
            candidate_set = set(candidate_block)
            candidate_ts = set()
            for ts in itertools.combinations(candidate_block, t):
                candidate_ts.add(tuple(sorted(ts)))
            
            # Count how many t-subsets from bi and bj are covered by candidate
            covered_from_bi = 0
            covered_from_bj = 0
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if ts in candidate_ts:
                    covered_from_bi += 1
            for ts in itertools.combinations(blocks[bj], t):
                ts = tuple(sorted(ts))
                if ts in candidate_ts:
                    covered_from_bj += 1
            
            # Check if removing bi and bj and adding candidate_block is valid
            # We need to ensure no t-subset becomes uncovered
            safe = True
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                if coverage_count.get(ts, 0) <= 1 and ts not in candidate_ts:
                    safe = False
                    break
            if safe:
                for ts in itertools.combinations(blocks[bj], t):
                    ts = tuple(sorted(ts))
                    if coverage_count.get(ts, 0) <= 1 and ts not in candidate_ts:
                        safe = False
                        break
            
            if safe:
                # Perform the swap
                # Remove bi and bj
                for ts in itertools.combinations(blocks[bi], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                for ts in itertools.combinations(blocks[bj], t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] -= 1
                    if coverage_count[ts] <= 0:
                        del coverage_count[ts]
                
                # Remove the blocks
                blocks.pop(max(bi, bj))
                blocks.pop(min(bi, bj))
                
                # Add the new block
                blocks.append(candidate_block)
                for ts in itertools.combinations(candidate_block, t):
                    ts = tuple(sorted(ts))
                    coverage_count[ts] = coverage_count.get(ts, 0) + 1
                
                improved = True
                break
    
    return blocks


def construct(v, k, t):
    global _global_deadline
    _global_deadline = time.time() + 110  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')
    
    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, _global_deadline)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < _global_deadline - 60 and restarts < 5:
        blocks = _greedy_cover(v, k, t, rng, _global_deadline - 60)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count (redundancy removal)
    if time.time() < _global_deadline - 20:
        best_blocks = _local_search(best_blocks, v, k, t, _global_deadline - 20)

    # Phase 3: Block swap search (more aggressive local search)
    if time.time() < _global_deadline - 5:
        best_blocks = _block_swap_search(best_blocks, v, k, t, _global_deadline - 2, rng)

    return best_blocks
# EVOLVE-BLOCK-END
