"""Covering design C(v,k,t) via optimized randomized greedy with local search."""
import itertools
import random
import sys
import numpy as np
from scipy import sparse

def construct(v, k, t):
    # Use optimized sparse greedy with multiple restarts and local search
    return _optimized_cover(v, k, t)


def _optimized_cover(v, k, t, num_restarts=10, seed=42):
    """
    Optimized covering design solver:
    1. Precompute all k-subsets and t-subsets.
    2. Build a sparse incidence matrix.
    3. Run multiple randomized greedy passes using sparse matrix-vector multiplication.
    4. Apply a redundancy removal pass to the best result.
    """
    rng = random.Random(seed)
    
    # Precompute subsets
    all_ksubsets = list(itertools.combinations(range(v), k))
    tsubsets = list(itertools.combinations(range(v), t))
    n_tsubs = len(tsubsets)
    n_ksubs = len(all_ksubsets)
    
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubsets)}
    
    # Build sparse incidence matrix
    # Rows: k-subsets, Cols: t-subsets
    row_indices = []
    col_indices = []
    for i, ksub in enumerate(all_ksubsets):
        for tsub in itertools.combinations(ksub, t):
            j = tsub_to_idx[tsub]
            row_indices.append(i)
            col_indices.append(j)
            
    incidence = sparse.csr_matrix(
        (np.ones(len(row_indices), dtype=np.int32), (row_indices, col_indices)),
        shape=(n_ksubs, n_tsubs)
    )
    
    best_blocks = None
    best_len = float('inf')
    
    for r in range(num_restarts):
        current_seed = seed + r
        rng = random.Random(current_seed)
        
        uncovered = np.ones(n_tsubs, dtype=np.int32)
        blocks = []
        
        while uncovered.any():
            # Calculate coverage of each block for currently uncovered t-subsets
            # sparse matrix @ dense vector
            covered_counts = incidence.dot(uncovered)
            
            # Find the best block(s)
            max_count = covered_counts.max()
            if max_count == 0:
                break
            
            # Randomized tie-breaking: pick randomly from blocks with max_count
            # To avoid finding all indices every time (expensive), we can just pick argmax
            # but for better exploration, let's pick from top candidates.
            # However, for speed, simple argmax with random restarts is often sufficient.
            # Let's stick to argmax but with random initial state? No, state is deterministic.
            # Let's add a small random noise to break ties?
            
            # Find all indices with max_count
            # This might be slow if done naively.
            # covered_counts is a 1D array.
            # indices = np.where(covered_counts == max_count)[0]
            # chosen_idx = rng.choice(indices)
            
            # Optimization: Just use argmax. The randomness comes from the restarts
            # and potentially the order if we shuffle?
            # Actually, pure greedy is deterministic.
            # To introduce randomness, we can shuffle the order of blocks in the matrix?
            # Or, we can pick randomly among the top 10.
            
            # Let's try picking randomly among all blocks with max_count.
            # Since max_count is usually unique or close, this is fast.
            top_indices = np.where(covered_counts == max_count)[0]
            chosen_idx = rng.choice(top_indices)
            
            blocks.append(list(all_ksubsets[chosen_idx]))
            
            # Update uncovered
            # uncovered[incidence[chosen_idx].toarray().flatten() == 1] = 0
            # Better:
            covered_cols = incidence[chosen_idx].indices
            uncovered[covered_cols] = 0
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
            # Early stopping if we hit the lower bound?
            # Lower bound for C(19,7,4) is 131.
            if best_len <= 131:
                break

    if best_blocks is None:
        # Fallback if something went wrong
        return _basic_greedy(v, k, t)
        
    # Local Search: Redundancy Removal
    best_blocks = _remove_redundant_blocks(best_blocks, v, k, t)
    
    return best_blocks


def _remove_redundant_blocks(blocks, v, k, t):
    """
    Remove blocks that are redundant (i.e., their t-subsets are covered by other blocks).
    """
    if not blocks:
        return blocks
    
    # Map each t-subset to the set of block indices that cover it
    tsub_to_blocks = {}
    for i, block in enumerate(blocks):
        for tsub in itertools.combinations(block, t):
            if tsub not in tsub_to_blocks:
                tsub_to_blocks[tsub] = set()
            tsub_to_blocks[tsub].add(i)
    
    # Iterate through blocks and try to remove them
    # We need to be careful: removing a block might make another block redundant.
    # So we repeat until no change.
    
    changed = True
    active_blocks = list(range(len(blocks)))
    
    while changed:
        changed = False
        # Sort by "criticality"? Or just iterate?
        # Iterating in order is fine.
        for i in list(active_blocks):
            # Check if block i is redundant
            is_redundant = True
            for tsub in itertools.combinations(blocks[i], t):
                covering_blocks = tsub_to_blocks.get(tsub, set())
                # If there is no other active block covering this t-subset, block i is essential
                if not covering_blocks - {i}:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove block i
                active_blocks.remove(i)
                changed = True
                # Update tsub_to_blocks
                for tsub in itertools.combinations(blocks[i], t):
                    if i in tsub_to_blocks[tsub]:
                        tsub_to_blocks[tsub].remove(i)
    
    # Return the actual blocks
    return [blocks[i] for i in active_blocks]


def _basic_greedy(v, k, t):
    """Fallback basic greedy."""
    all_tsubsets = set(itertools.combinations(range(v), t))
    uncovered = set(all_tsubsets)
    blocks = []
    
    while uncovered:
        seed_subset = next(iter(uncovered))
        block = list(seed_subset)
        
        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(combo + (x,)))
                    if new_tsub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
        
        block.sort()
        blocks.append(block)
        for tsub in itertools.combinations(block, t):
            uncovered.discard(tsub)
    
    return blocks
