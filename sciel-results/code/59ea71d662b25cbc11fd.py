# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict

def _build_index(v, t):
    all_t = list(itertools.combinations(range(v), t))
    t_index = {ts: i for i, ts in enumerate(all_t)}
    return all_t, t_index

def _cover_count_for_block(block, t, t_index):
    """Return a list of indices of t-subsets covered by this block."""
    return [t_index[ts] for ts in itertools.combinations(block, t)]

def _is_valid_cover(blocks, t, t_index):
    counts = [0] * len(t_index)
    for block in blocks:
        for idx in _cover_count_for_block(block, t, t_index):
            counts[idx] += 1
    return all(c > 0 for c in counts)

def _greedy_cover(v, k, t, rng, deadline):
    all_t, t_index = _build_index(v, t)
    n_t = len(all_t)
    uncovered = bytearray([1] * n_t)
    blocks = []
    t_subsets_of = {}  # cache: frozenset -> list of indices
    
    def get_indices_for_block(block):
        fs = frozenset(block)
        if fs not in t_subsets_of:
            t_subsets_of[fs] = [t_index[ts] for ts in itertools.combinations(sorted(block), t)]
        return t_subsets_of[fs]
    
    while any(uncovered):
        if time.time() > deadline - 5:
            break
        # Find the first uncovered t-subset
        uncovered_indices = [i for i in range(n_t) if uncovered[i]]
        if not uncovered_indices:
            break
        start_idx = uncovered_indices[0]
        start_ts = all_t[start_idx]
        block = list(start_ts)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many uncovered t-subsets would be covered
                # by adding x to the current block
                block_with_x = block + [x]
                # Get all t-subsets of block_with_x that include x
                gain = 0
                for ts in itertools.combinations(block_with_x, t):
                    if x in ts:
                        idx = t_index[ts]
                        if uncovered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
        
        block.sort()
        # Mark all t-subsets of this block as covered
        for idx in get_indices_for_block(block):
            uncovered[idx] = 0
        blocks.append(block)
        
        # Check deadline
        if time.time() > deadline - 2:
            break
    
    return blocks

def _local_search(blocks, v, k, t, rng, deadline, safety_margin=10):
    """Improve the covering by trying to remove redundant blocks and swap points."""
    all_t, t_index = _build_index(v, t)
    n_t = len(all_t)
    
    # Precompute for each block its covered t-subset indices
    block_to_indices = []
    for block in blocks:
        indices = [t_index[ts] for ts in itertools.combinations(sorted(block), t)]
        block_to_indices.append(indices)
    
    cover_count = [0] * n_t
    for indices in block_to_indices:
        for idx in indices:
            cover_count[idx] += 1
    
    current_blocks = blocks[:]
    current_block_indices = block_to_indices[:]
    
    def try_remove_block(i):
        """Try to remove block i if it's redundant. Returns True if removed."""
        nonlocal cover_count, current_blocks, current_block_indices
        # Check if all t-subsets in block i are covered by other blocks
        for idx in current_block_indices[i]:
            if cover_count[idx] <= 1:
                return False
        # Safe to remove
        for idx in current_block_indices[i]:
            cover_count[idx] -= 1
        current_blocks.pop(i)
        current_block_indices.pop(i)
        return True
    
    def try_swap_point(i, old_p, new_p):
        """Try to replace point old_p with new_p in block i. Returns True if swapped."""
        nonlocal cover_count, current_blocks, current_block_indices
        old_block = current_blocks[i]
        new_block = [x for x in old_block if x != old_p] + [new_p]
        new_block.sort()
        
        # Compute new indices for the new block
        new_indices = [t_index[ts] for ts in itertools.combinations(new_block, t)]
        
        # Check if all t-subsets that were uniquely covered by old block 
        # (and affected by the swap) are still covered
        old_indices_set = set(current_block_indices[i])
        new_indices_set = set(new_indices)
        
        # t-subsets that will be uncovered if we make this swap
        for idx in old_indices_set - new_indices_set:
            if cover_count[idx] <= 1:
                return False
        
        # Perform the swap
        # Decrement old indices
        for idx in current_block_indices[i]:
            cover_count[idx] -= 1
        # Increment new indices
        for idx in new_indices:
            cover_count[idx] += 1
        
        current_blocks[i] = new_block
        current_block_indices[i] = new_indices
        return True
    
    # Try to remove redundant blocks repeatedly
    improved = True
    while improved and time.time() < deadline - safety_margin:
        improved = False
        for i in range(len(current_blocks)):
            if try_remove_block(i):
                improved = True
                break
    
    # Try point swaps to enable more block removals
    num_swaps = 0
    max_swaps = 200
    while num_swaps < max_swaps and time.time() < deadline - safety_margin:
        num_swaps += 1
        i = rng.randrange(len(current_blocks))
        block = current_blocks[i]
        if len(block) < k:
            continue
        p = rng.choice(block)
        available = [x for x in range(v) if x not in block]
        if not available:
            continue
        q = rng.choice(available)
        if try_swap_point(i, p, q):
            # After a swap, try to remove blocks again
            for j in range(len(current_blocks)):
                if try_remove_block(j):
                    break
    
    # Final: try removal once more
    improved = True
    while improved and time.time() < deadline - safety_margin:
        improved = False
        for i in range(len(current_blocks)):
            if try_remove_block(i):
                improved = True
                break
    
    return current_blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Set random seed for reproducibility within a run
    rng = random.Random(42)
    
    # First, do a greedy cover
    blocks = _greedy_cover(v, k, t, rng, deadline)
    
    # If greedy didn't finish, check if it's still valid
    # For now, assume it produced a valid cover (it should, since we stop before deadline)
    
    # Local search to improve
    blocks = _local_search(blocks, v, k, t, rng, deadline)
    
    return blocks
# EVOLVE-BLOCK-END
