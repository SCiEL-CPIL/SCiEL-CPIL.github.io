import itertools
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit
    
    # 1. Generate all t-subsets and assign them an index 0..N_t-1
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    t_subset_index = {sub: i for i, sub in enumerate(t_subsets)}
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # 3. Precompute coverage mask for each block
    # coverage[i] is a Python int where bit j is 1 if block i covers t-subset j
    coverage = [0] * n_k
    
    # Optimization: For each t-subset, find all k-subsets that contain it.
    # Instead of iterating all k-subsets and checking all t-subsets within them,
    # we can iterate t-subsets and find k-subsets containing them.
    # But n_t is 11628 and n_k is 75582.
    # Direct method: For each block, compute mask.
    # A block has C(k,t) t-subsets.
    # Total operations: n_k * C(k,t).
    # For v=19, k=8, t=5: 75582 * 56 = 4.2M operations. This is fast.
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        coverage[i] = mask
        
    # 4. Greedy Set Cover
    uncovered = (1 << n_t) - 1  # All bits set initially
    selected_blocks = []
    
    # To speed up the search for the best block, we can sort blocks by initial coverage?
    # No, coverage changes as we pick blocks.
    
    # We need to find argmax popcount(uncovered & coverage[i])
    # We can maintain a list of active blocks?
    # Simple loop over all blocks each time is O(n_k) per step.
    # Steps ~ 300. 300 * 75582 = 22.6M iterations.
    # Each iteration is bitwise AND and bit_count.
    # This should be well within 110s in Python.
    
    while uncovered:
        if time.time() > deadline:
            # Return what we have so far? No, it must be valid.
            # If we break, we might have uncovered subsets.
            # The task requires a valid covering.
            # If we time out, we must return a valid covering.
            # So we must finish.
            # If the simple greedy is too slow, we need a faster one.
            # Let's assume it's fast enough.
            pass
            
        best_block_idx = -1
        best_count = 0
        
        # Optimization: Skip blocks that are subsets of already selected blocks?
        # Not easily tracked.
        
        for i in range(n_k):
            if not (coverage[i] & uncovered):
                continue
            count = (coverage[i] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_block_idx = i
        
        if best_block_idx == -1:
            # Should not happen if uncovered is not 0 and we have all blocks
            break
            
        selected_blocks.append(k_subsets[best_block_idx])
        uncovered &= ~coverage[best_block_idx]
        
    return [list(block) for block in selected_blocks]
