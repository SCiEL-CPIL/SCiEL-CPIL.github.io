"""
Covering design C(v, k, t) solver.
Strategy:
1. Fast greedy construction to get an initial covering.
2. Local search to remove redundant blocks.
3. Randomized restarts with different point orderings to find better covers.
4. Time-bounded search with deadline guard.
"""

import itertools
import time
import random
import sys

def _time_remaining(deadline):
    return deadline - time.time()

def _greedy_construct(v, k, t, point_order=None, deadline=None):
    """
    Greedy covering construction.
    Returns a list of blocks (each a sorted list of k integers).
    """
    if point_order is None:
        point_order = list(range(v))
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    if not uncovered:
        return blocks
    
    # For each block, we need to cover t-subsets
    # Greedy: pick the first uncovered t-subset, build a block around it
    while uncovered:
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset (deterministic for reproducibility within a run)
        # But we can randomize which uncovered subset to start with
        if uncovered:
            first_subset = random.choice(list(uncovered))
        else:
            break
        
        block = set(first_subset)
        
        # Grow the block to size k by adding points that cover the most uncovered t-subsets
        while len(block) < k:
            if deadline is not None and time.time() > deadline:
                break
            
            best_point = None
            best_gain = -1
            
            for x in point_order:
                if x in block:
                    continue
                
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and best_point is not None:
                    # Tie-breaking: prefer points that appear earlier in point_order
                    # But we already iterate in point_order, so first found wins
                    pass
            
            if best_point is None:
                break
            block.add(best_point)
        
        if len(block) == k:
            block_list = sorted(block)
            blocks.append(block_list)
            # Remove all t-subsets of this block from uncovered
            for ts in itertools.combinations(block_list, t):
                uncovered.discard(ts)
    
    return blocks

def _is_valid_covering(v, k, t, blocks):
    """Check if every t-subset is covered by at least one block."""
    all_t_subsets = set(itertools.combinations(range(v), t))
    covered = set()
    for block in blocks:
        for ts in itertools.combinations(block, t):
            covered.add(ts)
    return all_t_subsets.issubset(covered)

def _remove_redundant_blocks(v, k, t, blocks, deadline=None):
    """
    Try to remove blocks that are redundant (all their t-subsets are covered by other blocks).
    Returns a new list of blocks.
    """
    if not blocks:
        return blocks
    
    # For each t-subset, count how many blocks cover it
    # And for each block, track which t-subsets it covers
    from collections import defaultdict
    
    # Map each t-subset to the list of block indices that cover it
    ts_to_blocks = defaultdict(list)
    block_tsubsets = []
    
    for bi, block in enumerate(blocks):
        bts = list(itertools.combinations(block, t))
        block_tsubsets.append(bts)
        for ts in bts:
            ts_to_blocks[ts].append(bi)
    
    n_blocks = len(blocks)
    # active[bi] = True if block bi is still in the covering
    active = [True] * n_blocks
    
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        
        for bi in range(n_blocks):
            if not active[bi]:
                continue
            
            # Check if all t-subsets of block bi are covered by other active blocks
            redundant = True
            for ts in block_tsubsets[bi]:
                # Is ts covered by any other active block?
                found_other = False
                for bj in ts_to_blocks[ts]:
                    if bj != bi and active[bj]:
                        found_other = True
                        break
                if not found_other:
                    redundant = False
                    break
            
            if redundant:
                active[bi] = False
                changed = True
    
    result = [blocks[bi] for bi in range(n_blocks) if active[bi]]
    return result

def _random_point_order(v, rng=None):
    """Generate a random permutation of points."""
    if rng is None:
        rng = random
    order = list(range(v))
    rng.shuffle(order)
    return order

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a sorted list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    # Deadline: leave 5 seconds safety margin before the 120s limit
    deadline = start_time + 115.0
    
    best_blocks = None
    best_len = float('inf')
    
    # First, try the default greedy
    blocks = _greedy_construct(v, k, t, point_order=list(range(v)), deadline=deadline)
    if blocks:
        # Try to remove redundant blocks
        reduced = _remove_redundant_blocks(v, k, t, blocks, deadline=deadline)
        if len(reduced) < len(blocks):
            blocks = reduced
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]
    
    # Randomized restarts
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline - 2.0:
        attempt += 1
        point_order = _random_point_order(v, rng)
        blocks = _greedy_construct(v, k, t, point_order=point_order, deadline=deadline)
        if blocks:
            reduced = _remove_redundant_blocks(v, k, t, blocks, deadline=deadline)
            if len(reduced) < len(blocks):
                blocks = reduced
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]
                # If we found a very good solution, we can stop early
                # The theoretical lower bound for C(24,7,3) is 69
                if best_len <= 76:
                    # Continue searching for a few more iterations to see if we can do better
                    pass
    
    if best_blocks is None:
        # Fallback: return a trivial covering (should not happen for valid inputs)
        return []
    
    return best_blocks
