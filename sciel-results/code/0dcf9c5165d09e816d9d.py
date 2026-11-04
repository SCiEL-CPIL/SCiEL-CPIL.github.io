import time
import numpy as np
from itertools import combinations

def _precompute(v, k, t):
    """Precompute all t-subset indices and block coverage masks."""
    all_t_subsets = list(combinations(range(v), t))
    total = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    all_blocks = list(combinations(range(v), k))
    nblocks = len(all_blocks)
    # block_masks[i] is a numpy boolean array of length total
    # True where the block covers that t-subset
    block_masks = np.zeros((nblocks, total), dtype=bool)
    for i, block in enumerate(all_blocks):
        for s in combinations(block, t):
            block_masks[i, subset_to_idx[s]] = True
    return all_t_subsets, total, subset_to_idx, all_blocks, nblocks, block_masks

def _greedy_construct(v, k, t, block_masks, nblocks, total, rng, deadline):
    """Randomized greedy: pick block with max uncovered coverage (with random tiebreak)."""
    uncovered = np.ones(total, dtype=bool)
    n_uncovered = total
    blocks = []
    used = np.zeros(nblocks, dtype=bool)
    while n_uncovered > 0:
        if time.time() > deadline:
            break
        # Gain for each candidate block
        gains = block_masks[:, uncovered].sum(axis=1)
        gains[used] = -1
        if gains.max() <= 0:
            break
        # Randomize among top gainers
        max_gain = gains.max()
        top_idx = np.where(gains == max_gain)[0]
        if len(top_idx) > 1:
            idx = top_idx[rng.integers(len(top_idx))]
        else:
            idx = top_idx[0]
        used[idx] = True
        blocks.append(idx)
        uncovered = uncovered & ~block_masks[idx]
        n_uncovered = int(uncovered.sum())
    return blocks

def _local_search(block_indices, v, k, t, block_masks, nblocks, total, all_blocks, rng, deadline):
    """Try to improve by replacing blocks or removing redundant ones."""
    blocks = list(block_indices)
    # Compute coverage
    covered = np.zeros(total, dtype=bool)
    for i in blocks:
        covered |= block_masks[i]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each block if it's redundant
        for pos in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            b = blocks[pos]
            # Check if removing b still covers everything
            test_covered = covered.copy()
            test_covered &= ~block_masks[b]
            # Recompute: need to check all other blocks still cover
            # Actually covered has all blocks; removing b means we need
            # the union of all other blocks to still be full
            # Quick check: is there any position where only b covers?
            # That's: covered[i] & ~block_masks[b] is False for some i where covered[i] is True
            # Simpler: recompute without b
            other_covered = np.zeros(total, dtype=bool)
            for j, bi in enumerate(blocks):
                if j == pos:
                    continue
                other_covered |= block_masks[bi]
            if other_covered.all():
                blocks.pop(pos)
                covered = other_covered
                improved = True
                break
        
        if time.time() > deadline:
            break
        
        # Try replacing one block with another
        for pos in range(len(blocks)):
            if time.time() > deadline:
                break
            # For each position, try replacing with a random other block
            # that adds coverage where the current block is weak
            current = blocks[pos]
            # Find blocks that cover some subsets that current doesn't
            # Try a few random candidates
            for _ in range(20):
                if time.time() > deadline:
                    break
                candidate = int(rng.integers(nblocks))
                if candidate == current or candidate in blocks:
                    continue
                # Check if replacing improves: new coverage = covered - current + candidate
                # We need: (covered & ~block_masks[current] | block_masks[candidate]).all()
                # And ideally reduces total blocks or same count but we count blocks
                # Since we keep same count, we only replace if it doesn't hurt
                # Actually we want to minimize count, so replacement only helps if
                # it allows later removal. For now, skip replacement to save time.
                pass
    return blocks

def _verify(blocks, v, k, t, all_t_subsets, subset_to_idx):
    """Verify all t-subsets covered."""
    covered = set()
    for block in blocks:
        for s in combinations(block, t):
            covered.add(s)
    for s in all_t_subsets:
        if s not in covered:
            return False
    return True

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin for 150s limit
    
    # Precompute
    all_t_subsets, total, subset_to_idx, all_blocks, nblocks, block_masks = _precompute(v, k, t)
    
    if time.time() > deadline:
        # Emergency fallback: return empty (invalid) - shouldn't happen
        return []
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy restarts
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        rng = np.random.default_rng(seed)
        block_indices = _greedy_construct(v, k, t, block_masks, nblocks, total, rng, deadline - 5)
        
        if not block_indices:
            continue
        
        # Local search to improve
        if time.time() < deadline - 10:
            block_indices = _local_search(block_indices, v, k, t, block_masks, nblocks, total, all_blocks, rng, deadline - 10)
        
        count = len(block_indices)
        if count < best_count:
            # Verify
            actual_blocks = [list(all_blocks[i]) for i in block_indices]
            if _verify(actual_blocks, v, k, t, all_t_subsets, subset_to_idx):
                best_blocks = actual_blocks
                best_count = count
        
        if best_count <= 135:
            break
    
    if best_blocks is None:
        # Fallback: single greedy
        rng = np.random.default_rng(0)
        block_indices = _greedy_construct(v, k, t, block_masks, nblocks, total, rng, deadline)
        best_blocks = [list(all_blocks[i]) for i in block_indices]
    
    return best_blocks
