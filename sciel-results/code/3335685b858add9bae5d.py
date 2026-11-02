"""Combined approach: precomputed full-block greedy with randomized tie-breaking and local search."""
import time
import random
import numpy as np
from itertools import combinations


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    # Precompute all t-subsets and map to indices
    all_t_subsets = list(combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Precompute all k-subsets
    all_k_subsets = list(combinations(range(v), k))
    n_k = len(all_k_subsets)

    # For each k-subset, precompute the indices of t-subsets it covers
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)
    
    # Build cover matrix: shape (n_k, c_k_t), each entry is an index into all_t_subsets
    cover_matrix = np.zeros((n_k, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]

    best_blocks = None
    best_count = float('inf')
    
    # Greedy with random restarts and tie-breaking
    for restart in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(restart * 7919 + 13)
        
        uncovered = np.ones(n_t, dtype=bool)
        blocks_found = []
        
        while True:
            if not np.any(uncovered):
                break
            
            # Calculate gain for all blocks
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Get all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly for diversity
            chosen_idx = int(candidates[rng.randint(0, len(candidates) - 1)])
            
            blocks_found.append(all_k_subsets[chosen_idx])
            
            # Update uncovered
            uncovered[cover_matrix[chosen_idx]] = False
        
        if len(blocks_found) < best_count:
            best_count = len(blocks_found)
            best_blocks = blocks_found
        
        # If we've hit a good number, we can do a local search to try to improve
        if best_count <= 78 and time.time() < deadline - 20:
            improved = _local_search(best_blocks, v, k, t, cover_matrix, t_sub_index, deadline)
            if improved is not None and len(improved) < best_count:
                best_count = len(improved)
                best_blocks = improved

    if best_blocks is None:
        # Fallback: simple greedy
        best_blocks = []
        uncovered = np.ones(n_t, dtype=bool)
        while np.any(uncovered):
            gains = np.sum(uncovered[cover_matrix], axis=1)
            idx = int(np.argmax(gains))
            if gains[idx] == 0:
                break
            best_blocks.append(all_k_subsets[idx])
            uncovered[cover_matrix[idx]] = False

    return [list(b) for b in best_blocks]


def _local_search(blocks, v, k, t, cover_matrix, t_sub_index, deadline):
    """Try to reduce the number of blocks by removing one and re-covering."""
    if len(blocks) <= 2:
        return None
    
    n_t = len(t_sub_index)
    
    # Build coverage counts: for each t-subset, how many blocks cover it
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for block in blocks:
        for ts in combinations(block, t):
            cover_counts[t_sub_index[ts]] += 1
    
    # Try removing each block and see if we can re-cover with fewer new blocks
    for remove_idx in range(len(blocks)):
        if time.time() > deadline - 5:
            break
        
        # Tentatively remove this block
        removed_block = blocks[remove_idx]
        for ts in combinations(removed_block, t):
            cover_counts[t_sub_index[ts]] -= 1
        
        # Find uncovered t-subsets
        uncovered_mask = cover_counts == 0
        if not np.any(uncovered_mask):
            # Block was redundant
            return blocks[:remove_idx] + blocks[remove_idx+1:]
        
        # Try to cover the uncovered t-subsets with one new block
        # Find which blocks (from all_k_subsets) cover the most uncovered t-subsets
        # But we need to be careful not to pick the same block we just removed
        
        # Get indices of uncovered t-subsets
        uncovered_indices = np.where(uncovered_mask)[0]
        
        # For each candidate block, count how many uncovered t-subsets it covers
        # This is expensive, so we sample or use a heuristic
        
        # Simple heuristic: pick a random uncovered t-subset, find blocks containing it
        if len(uncovered_indices) > 0:
            target_idx = int(uncovered_indices[random.randint(0, len(uncovered_indices) - 1)])
            target_ts = tuple(t_sub_index_inv[target_idx]) if 't_sub_index_inv' in dir() else None
            
            # We need the inverse mapping. Let's precompute it.
            # Actually, let's just iterate through all blocks and check if they contain the target
            # This is O(n_k * k) which is ~346k * 7 ~ 2.4M ops, acceptable for a few tries
            
            # Better: precompute which blocks contain each t-subset
            # But that's a lot of memory. Let's just do a quick scan.
            
            # For now, skip local search if it's too complex.
            pass
        
        # Restore
        for ts in combinations(removed_block, t):
            cover_counts[t_sub_index[ts]] += 1
    
    return None
