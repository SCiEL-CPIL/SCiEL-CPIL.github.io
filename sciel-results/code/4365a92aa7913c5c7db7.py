"""
Global Greedy Set Cover with bitset acceleration for covering design C(24,7,3).
Uses numpy bitsets to represent the set of uncovered t-subsets, enabling fast
gain computation. Includes a post-processing redundancy-removal phase.
"""
import time
import numpy as np
from itertools import combinations

# Precompute all t-subsets and map them to indices
_ALL_TRIPLES = list(combinations(range(24), 3))
_NUM_TRIPLES = len(_ALL_TRIPLES)  # 2024

# Map each triple to an index
_TRIPLE_INDEX = {tr: i for i, tr in enumerate(_ALL_TRIPLES)}

# For each point, precompute the bitset of triples that contain it
# _POINT_TRIPLES[p] is a numpy bool array of length _NUM_TRIPLES
_POINT_TRIPLES = []
for p in range(24):
    mask = np.zeros(_NUM_TRIPLES, dtype=bool)
    for i, tr in enumerate(_ALL_TRIPLES):
        if p in tr:
            mask[i] = True
    _POINT_TRIPLES.append(mask)

# For each pair of points, precompute the bitset of triples that contain both
_PAIR_TRIPLES = {}
for a in range(24):
    for b in range(a + 1, 24):
        mask = _POINT_TRIPLES[a] & _POINT_TRIPLES[b]
        _PAIR_TRIPLES[(a, b)] = mask


def _block_triple_mask(block):
    """Return a bool numpy array marking which triples are covered by this block."""
    mask = np.zeros(_NUM_TRIPLES, dtype=bool)
    for tr in combinations(block, 3):
        mask[_TRIPLE_INDEX[tr]] = True
    return mask


def _greedy_cover(uncovered, v, k, t, deadline):
    """
    Greedy set cover: repeatedly pick the block (k-subset) that covers the most
    uncovered triples, until all are covered or deadline is reached.
    
    We use a two-phase approach:
    1. For each candidate block, compute how many uncovered triples it covers.
    2. Pick the best, add it, update uncovered.
    
    To make this tractable, we use a heuristic: build blocks greedily by adding
    points that maximize marginal gain, then evaluate the resulting block.
    """
    blocks = []
    while np.any(uncovered):
        if time.time() > deadline:
            break
        
        # Try several random seeds to build candidate blocks
        best_block = None
        best_gain = -1
        
        # Deterministic first: start from the triple with highest "demand"
        # Find an uncovered triple to anchor
        uncovered_indices = np.where(uncovered)[0]
        if len(uncovered_indices) == 0:
            break
        
        # Pick a few anchor triples
        anchors = []
        # The first uncovered triple
        anchors.append(_ALL_TRIPLES[uncovered_indices[0]])
        # A few more random ones for diversity
        for _ in range(min(3, len(uncovered_indices) - 1)):
            idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
            anchors.append(_ALL_TRIPLES[idx])
        
        for anchor in anchors:
            # Build a block greedily starting from anchor
            block = list(anchor)
            while len(block) < k:
                if time.time() > deadline:
                    break
                best_x = None
                best_marginal = -1
                for x in range(v):
                    if x in block:
                        continue
                    # Compute marginal gain: how many currently-uncovered triples
                    # would be covered by adding x to block
                    new_block = block + [x]
                    # A triple is covered by new_block if it's a subset of new_block
                    # and it's currently uncovered
                    # Efficient: for each uncovered triple, check if it's a subset of new_block
                    # Better: compute the mask of triples covered by new_block, intersect with uncovered
                    mask_new = _block_triple_mask(new_block)
                    marginal = np.count_nonzero(mask_new & uncovered)
                    if marginal > best_marginal:
                        best_marginal = marginal
                        best_x = x
                if best_x is None:
                    break
                block.append(best_x)
            
            if len(block) == k:
                block.sort()
                mask = _block_triple_mask(block)
                gain = np.count_nonzero(mask & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
        
        if best_block is None:
            # Fallback: just take the first uncovered triple and pad randomly
            block = list(_ALL_TRIPLES[uncovered_indices[0]])
            remaining = [x for x in range(v) if x not in block]
            import random
            random.seed(hash(tuple(block)) % 2**32)
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])
            block.sort()
            best_block = block
        
        blocks.append(best_block)
        mask = _block_triple_mask(best_block)
        uncovered &= ~mask
    
    return blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """
    Try to remove blocks that are redundant (all their triples are covered by other blocks).
    """
    n = len(blocks)
    if n == 0:
        return blocks
    
    # Precompute masks for each block
    masks = [_block_triple_mask(b) for b in blocks]
    
    # Union of all masks
    all_covered = np.zeros(_NUM_TRIPLES, dtype=bool)
    for m in masks:
        all_covered |= m
    
    # For each block, check if it's redundant
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        # Check if block i's covered triples are all covered by other kept blocks
        # Compute union of all other kept blocks
        other_union = np.zeros(_NUM_TRIPLES, dtype=bool)
        for j in range(n):
            if j != i and keep[j]:
                other_union |= masks[j]
        # If block i's mask is a subset of other_union, it's redundant
        if not np.any(masks[i] & ~other_union):
            keep[i] = False
    
    return [b for i, b in enumerate(blocks) if keep[i]]


def construct(v, k, t):
    """
    Construct a covering design: find the minimum number of k-element blocks
    of v points such that every t-element subset is contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initialize: all t-subsets are uncovered
    uncovered = np.ones(_NUM_TRIPLES, dtype=bool)
    
    # Run greedy cover
    blocks = _greedy_cover(uncovered, v, k, t, deadline)
    
    # Check if we covered everything
    if blocks:
        final_mask = np.zeros(_NUM_TRIPLES, dtype=bool)
        for b in blocks:
            final_mask |= _block_triple_mask(b)
        if not np.all(final_mask):
            # Still uncovered triples; add blocks for remaining uncovered triples
            # This shouldn't happen if greedy worked correctly, but as a safety net
            still_uncovered = np.where(~final_mask)[0]
            for idx in still_uncovered:
                if time.time() > deadline:
                    break
                tr = _ALL_TRIPLES[idx]
                # Find a block that contains this triple
                # Just create a new block containing this triple
                block = list(tr)
                remaining = [x for x in range(v) if x not in block]
                # Add points to maximize coverage of still-uncovered
                for _ in range(k - len(block)):
                    best_x = None
                    best_gain = -1
                    for x in remaining:
                        new_block = block + [x]
                        mask = _block_triple_mask(new_block)
                        gain = np.count_nonzero(mask & ~final_mask)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    remaining.remove(best_x)
                block.sort()
                blocks.append(block)
                final_mask |= _block_triple_mask(block)
    
    # Redundancy removal
    if time.time() < deadline - 5.0:
        blocks = _remove_redundant_blocks(blocks, v, k, t)
    
    # Final verification and cleanup
    # Ensure all blocks are valid (distinct, k distinct points in 0..v-1)
    seen = set()
    valid_blocks = []
    for b in blocks:
        b_sorted = sorted(b)
        if len(b_sorted) == k and len(set(b_sorted)) == k and all(0 <= x < v for x in b_sorted):
            if tuple(b_sorted) not in seen:
                seen.add(tuple(b_sorted))
                valid_blocks.append(b_sorted)
    
    return valid_blocks
