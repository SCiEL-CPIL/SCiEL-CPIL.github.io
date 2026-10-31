"""Rewritten program for covering design C(v,k,t) with a stronger greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t): a set of k-subsets (blocks) of {0..v-1}
    such that every t-subset is contained in at least one block.
    Minimize the number of blocks.
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # 1. Precompute all t-subsets and assign an index to each
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute the bitmask for every possible k-subset (block)
    # A bit i is set if the t-subset at index i is contained in the block.
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_blocks = len(all_k_subsets)
    
    # Build block_masks as a list of Python ints (bitmasks)
    block_masks = []
    for block in all_k_subsets:
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    def popcount(x):
        if hasattr(x, 'bit_count'):
            return x.bit_count()
        return bin(x).count('1')

    def run_greedy(seed):
        rng = random.Random(seed)
        unc = (1 << n_t) - 1
        sel = []
        sel_idx_set = set()
        
        while unc:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_indices = []
            
            for i, mask in enumerate(block_masks):
                if i in sel_idx_set:
                    continue
                x = unc & mask
                if x == 0:
                    continue
                gain = popcount(x)
                
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain and gain > 0:
                    best_indices.append(i)
            
            if not best_indices:
                break
            
            chosen_idx = rng.choice(best_indices)
            
            sel.append(all_k_subsets[chosen_idx])
            sel_idx_set.add(chosen_idx)
            unc &= ~block_masks[chosen_idx]
            
            if time.time() > deadline - 5.0:
                return None
        
        return sel

    def local_search(blocks, deadline):
        """Try to improve the covering by removing redundant blocks and adding new ones."""
        if not blocks:
            return blocks
        
        # Build coverage count: for each t-subset, how many blocks cover it
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        # Map block index to its mask
        block_mask_map = {}
        for b_idx, block in enumerate(blocks):
            mask = 0
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                mask |= (1 << idx)
            block_mask_map[b_idx] = mask
        
        # Count coverage
        for b_idx in range(len(blocks)):
            mask = block_mask_map[b_idx]
            # Add 1 to positions where mask has bits set
            # Use numpy for efficiency
            words = []
            m = mask
            while m:
                lsb = m & (-m)
                words.append(lsb.bit_length() - 1)
                m ^= lsb
            if words:
                cover_count[words] += 1
        
        improved = True
        current_blocks = list(blocks)
        
        while improved and time.time() < deadline:
            improved = False
            
            # Try to remove each block if it's redundant
            for b_idx in range(len(current_blocks)):
                if time.time() > deadline:
                    break
                
                mask = block_mask_map[b_idx]
                # Check if any t-subset would become uncovered
                # Get all t-subsets covered by this block
                words = []
                m = mask
                while m:
                    lsb = m & (-m)
                    words.append(lsb.bit_length() - 1)
                    m ^= lsb
                
                if not words:
                    continue
                
                # Check if all these t-subsets have cover_count > 1
                counts = cover_count[words]
                if np.all(counts > 1):
                    # Safe to remove
                    for w in words:
                        cover_count[w] -= 1
                    del current_blocks[b_idx]
                    # Rebuild block_mask_map
                    block_mask_map = {}
                    for new_idx, block in enumerate(current_blocks):
                        mask = 0
                        for ts in itertools.combinations(block, t):
                            idx = t_subset_index[ts]
                            mask |= (1 << idx)
                        block_mask_map[new_idx] = mask
                    improved = True
                    break  # Restart loop after modification
        
        return current_blocks

    best_blocks = None
    best_count = float('inf')
    
    # First run: deterministic (seed=0)
    res = run_greedy(0)
    if res:
        best_blocks = res
        best_count = len(res)
    
    # Try more seeds if time allows
    for seed in range(1, 50):
        if time.time() > deadline - 15.0:
            break
        res = run_greedy(seed)
        if res and len(res) < best_count:
            best_blocks = res
            best_count = len(res)
    
    # Local search: try to remove redundant blocks
    if best_blocks is not None and time.time() < deadline - 5.0:
        improved_blocks = local_search(best_blocks, deadline - 2.0)
        if len(improved_blocks) < best_count:
            best_blocks = improved_blocks
            best_count = len(best_blocks)

    if best_blocks is None:
        # Fallback: should not happen if first run succeeds
        return []

    return best_blocks
# EVOLVE-BLOCK-END
