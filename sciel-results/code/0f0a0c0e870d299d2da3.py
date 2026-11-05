import itertools
import numpy as np
import random

# Cache for precomputation to avoid recalculating on multiple calls if any
_cache = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _cache:
        return _cache[key]
    
    # Only efficient for small v, k, t
    if v > 22 or k > 12 or t > 6:
        return None

    # Map t-subsets to indices
    t_combs = list(itertools.combinations(range(v), t))
    t_to_idx = {comb: i for i, comb in enumerate(t_combs)}
    num_t = len(t_combs)
    num_words = (num_t + 63) // 64
    
    # Generate all k-subsets
    k_combs = list(itertools.combinations(range(v), k))
    num_k = len(k_combs)
    
    # Build masks
    # Using numpy for fast bitwise operations later
    masks = np.zeros((num_k, num_words), dtype=np.uint64)
    
    for i, k_comb in enumerate(k_combs):
        # Find all t-subsets within this k-subset
        for t_comb in itertools.combinations(k_comb, t):
            idx = t_to_idx[t_comb]
            word_idx = idx // 64
            bit_idx = idx % 64
            masks[i, word_idx] |= (1 << bit_idx)
            
    _cache[key] = (k_combs, masks, num_words, num_t)
    return _cache[key]

def construct(v, k, t):
    # Fallback for non-optimized sizes or if precomputation fails
    data = _precompute(v, k, t)
    if data is None:
        # Use the seed's greedy approach as fallback
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    k_combs, masks, num_words, num_t = data
    
    # Initial uncovered mask: all bits set
    uncovered = np.ones(num_words, dtype=np.uint64)
    if num_t % 64 != 0:
        # Mask out bits beyond num_t in the last word
        valid_bits = num_t % 64
        mask_last = (1 << valid_bits) - 1
        uncovered[-1] = mask_last
    
    best_blocks = None
    best_size = float('inf')
    
    # Number of restarts
    # Adjust based on time limit. 100 restarts is safe.
    num_restarts = 150
    
    for _ in range(num_restarts):
        current_uncovered = uncovered.copy()
        blocks = []
        
        while current_uncovered.any():
            # Calculate gains
            gains = np.bitwise_and(masks, current_uncovered).sum(axis=1)
            
            max_gain = gains.max()
            if max_gain == 0:
                # Should not happen if current_uncovered is not empty
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to break symmetry
            idx = candidates[random.randint(0, len(candidates) - 1)]
            
            blocks.append(list(k_combs[idx]))
            
            # Update uncovered
            current_uncovered &= ~masks[idx]
            
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
            # Early stop if we hit a theoretical lower bound or known best
            # Known best for C(19,8,5) is 335.
            if best_size <= 335:
                break

    return best_blocks
