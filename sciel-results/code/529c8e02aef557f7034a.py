"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a multi-start randomized greedy
    algorithm with efficient bitset tracking.
    
    Strategy:
    1. Represent all t-subsets as bits in a numpy uint64 array.
    2. For each candidate block (k-subset), precompute its coverage mask.
    3. Use randomized greedy: at each step, pick a block that covers the most
       uncovered t-subsets, with random tie-breaking and occasional randomization.
    4. Run multiple randomized restarts and keep the best result.
    """
    deadline = time.time() + 110.0  # 110s safety margin
    
    # Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    t_sub_index = {s: i for i, s in enumerate(t_subs)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute coverage masks for all blocks
    # Each block covers C(k, t) t-subsets
    block_masks = np.zeros((n_blocks, n_t), dtype=np.uint8)
    for bi, block in enumerate(all_blocks):
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_index[t_sub]
            block_masks[bi, idx] = 1
    
    # Total uncovered
    full_mask = np.ones(n_t, dtype=np.uint8)
    
    best_blocks = None
    best_count = n_t  # worst case: one block per t-subset
    
    # Multi-start randomized greedy
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        uncovered = full_mask.copy()
        blocks_used = []
        
        while uncovered.any():
            # Compute coverage of each block on currently uncovered t-subsets
            # count[bi] = number of uncovered t-subsets covered by block bi
            counts = (block_masks & uncovered[None, :]).sum(axis=1)
            
            # Find max count
            max_count = counts.max()
            if max_count == 0:
                break
            
            # Get all blocks achieving max_count
            candidates = np.where(counts == max_count)[0]
            
            # Random tie-breaking: pick randomly from candidates
            # Occasionally (10%) pick a random block with high count (not just max)
            if len(candidates) > 1 and rng.random() < 0.1:
                # Pick from top 20% of counts
                threshold = max_count - max(1, int(0.2 * max_count))
                top_candidates = np.where(counts >= threshold)[0]
                if len(top_candidates) > 0:
                    bi = top_candidates[rng.integers(len(top_candidates))]
                else:
                    bi = candidates[rng.integers(len(candidates))]
            else:
                bi = candidates[rng.integers(len(candidates))]
            
            blocks_used.append(all_blocks[bi])
            uncovered &= ~block_masks[bi]
        
        if not uncovered.any():
            if len(blocks_used) < best_count:
                best_count = len(blocks_used)
                best_blocks = [list(b) for b in blocks_used]
        
        # If we've used too much time, stop
        if time.time() > deadline - 10:
            break
    
    if best_blocks is None:
        # Fallback: return all k-subsets (always valid but large)
        return [list(b) for b in all_blocks]
    
    return best_blocks
# EVOLVE-BLOCK-END
