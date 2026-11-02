"""Covering design C(v,k,t): bitset-accelerated randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import Counter


def _all_t_subsets(v, t):
    """Return list of all t-subsets as sorted tuples."""
    return list(itertools.combinations(range(v), t))


def _subset_to_idx(t_subs):
    """Map each t-subset tuple to its index."""
    return {s: i for i, s in enumerate(t_subs)}


def _block_tsub_indices(block, t, idx_map):
    """Return set of indices of t-subsets covered by block."""
    return {idx_map[comb] for comb in itertools.combinations(block, t)}


def _greedy_pass(v, k, t, rng=None):
    """One greedy pass using bitmasks for speed. Returns list of blocks."""
    if rng is None:
        rng = random.Random()
    
    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)
    
    # Use a bitarray-like structure: list of booleans for uncovered status
    uncovered = [True] * n_tsubs
    uncovered_count = n_tsubs
    blocks = []
    
    # Precompute for each point x, the list of t-subsets that contain x
    # and for each (x, block_so_far), we need to compute gain efficiently
    # For now, use a simpler approach with caching
    
    while uncovered_count > 0:
        # Pick a random uncovered t-subset to start the block
        # Find an uncovered index
        while True:
            idx = rng.randrange(n_tsubs)
            if uncovered[idx]:
                break
        start_sub = t_subs[idx]
        block = list(start_sub)
        
        # Add points greedily
        while len(block) < k:
            block_set = set(block)
            candidates = [x for x in range(v) if x not in block_set]
            best_x = None
            best_gain = -1
            
            # Precompute t-1 subsets of current block
            t1_subs = list(itertools.combinations(block, t - 1))
            
            for x in candidates:
                gain = 0
                for rest in t1_subs:
                    # rest is a tuple of t-1 elements, add x
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in idx_map:
                        i = idx_map[new_sub]
                        if uncovered[i]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        
        # Remove covered t-subsets
        for comb in itertools.combinations(block, t):
            i = idx_map[comb]
            if uncovered[i]:
                uncovered[i] = False
                uncovered_count -= 1
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t, rng=None):
    """Try to remove redundant blocks from a valid covering using bitmasks."""
    if not blocks:
        return blocks
    if rng is None:
        rng = random.Random(0)
    
    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)
    
    # For each block, compute the set of t-subset indices it covers
    block_masks = []
    for b in blocks:
        mask = set()
        for comb in itertools.combinations(b, t):
            mask.add(idx_map[comb])
        block_masks.append(mask)
    
    n_blocks = len(blocks)
    
    # Count how many blocks cover each t-subset
    coverage_count = [0] * n_tsubs
    for mask in block_masks:
        for i in mask:
            coverage_count[i] += 1
    
    # Try to remove blocks in random order (multiple passes for better results)
    kept_indices = list(range(n_blocks))
    changed = True
    max_passes = 5
    
    for _ in range(max_passes):
        if not changed:
            break
        changed = False
        rng.shuffle(kept_indices)
        
        new_kept = []
        for i in kept_indices:
            # Check if block i is redundant: all its t-subsets are covered by other kept blocks
            redundant = True
            for tsub_idx in block_masks[i]:
                if coverage_count[tsub_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_kept.append(i)
            else:
                # Remove block i
                for tsub_idx in block_masks[i]:
                    coverage_count[tsub_idx] -= 1
                changed = True
        
        kept_indices = new_kept
    
    return [blocks[i] for i in kept_indices]


def construct(v, k, t):
    """Find a covering design using randomized greedy with redundancy removal."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin: 115s out of 150s limit
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple randomized greedy passes
    attempt = 0
    while time.time() < deadline:
        rng = random.Random(42 + attempt)
        blocks = _greedy_pass(v, k, t, rng)
        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, v, k, t, rng)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
        
        # If we've found a good solution, we can stop early
        if best_len <= 340:  # Close to known best of 335
            break
        
        attempt += 1
    
    if best_blocks is None:
        # Fallback: single deterministic pass
        best_blocks = _greedy_pass(v, k, t)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
    
    return best_blocks
# EVOLVE-BLOCK-END
