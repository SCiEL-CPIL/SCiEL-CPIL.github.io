import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 500 # Adjust based on speed
    
    # Helper to check if a set of block indices forms a valid cover
    # Returns True if valid, and updates the 'uncovered' mask if needed
    # For efficiency in local search, we maintain counts of how many times each t-subset is covered.
    
    def is_valid_cover(block_indices):
        if not block_indices:
            return num_t_subsets == 0
        # Count coverage
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            # Iterate set bits
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        for c in cover_counts:
            if c == 0:
                return False
        return True

    def prune_redundant(block_indices):
        """
        Remove blocks that are redundant.
        Returns a new list of block indices.
        """
        if not block_indices:
            return []
        
        # Count coverage
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                # Check if removing idx keeps all counts >= 1
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    # Remove idx
                    # Decrement counts
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i]) # Initially all uncovered
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Find max gain
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Random choice among max gain candidates
            chosen_idx = random.choice(candidates)
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0 # Safety

        if uncovered == 0:
            # Greedy solution found
            # Apply pruning
            pruned_blocks = prune_redundant(current_blocks)
            
            # Local Search: Try 1-swap to enable further pruning
            # This is expensive, so we limit it
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_size = len(pruned_blocks)
                # Try to swap each block
                for i in range(len(pruned_blocks)):
                    if time.time() > deadline:
                        break
                    idx_to_remove = pruned_blocks[i]
                    # Try adding a new block
                    # To save time, we only try a few random candidates or those with high overlap
                    # For now, let's just try random candidates from the pool
                    # This is O(N^2) which might be too slow.
                    # Alternative: Just do more restarts.
                    # Given the tight time, maybe skip complex local search and rely on restarts + pruning.
                    # The pruning itself is a significant improvement over raw greedy.
                    pass
            
            if len(pruned_blocks) < best_count:
                best_count = len(pruned_blocks)
                best_blocks = [list(k_subsets[i]) for i in pruned_blocks]

    if best_blocks is None:
        # Fallback
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        while uncovered:
            max_gain = 0
            best_block_idx = -1
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    best_block_idx = i
            if max_gain == 0:
                break
            current_blocks.append(best_block_idx)
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        
        pruned_blocks = prune_redundant(current_blocks)
        best_blocks = [list(k_subsets[i]) for i in pruned_blocks]

    return best_blocks
