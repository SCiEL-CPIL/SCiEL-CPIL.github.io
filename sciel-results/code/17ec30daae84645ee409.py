import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s (evaluator kills at 150s, but be safe)

    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Enumerate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # Compute masks: masks[i] is a bitmask where bit j is 1 if k_subset i covers t_subset j
    # For v=19, k=8, t=5: num_t_subsets = C(19,5) = 11628, num_k_subsets = C(19,8) = 75582
    # These are large integers (11628 bits), so we use Python's arbitrary precision ints
    
    # Build masks efficiently
    masks = [0] * num_k_subsets
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask

    # Precompute popcount helper
    def popcount(x):
        return x.bit_count()

    # Greedy initialization with randomization
    def greedy_init(randomized=True):
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        while uncovered:
            max_gain = 0
            candidates = []
            
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                gain = popcount(masks[i] & uncovered)
                if gain > max_gain:
                    max_gain = gain
                    candidates = [i]
                elif gain == max_gain and gain > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                break
            
            if randomized and candidates:
                # Pick randomly among top candidates for diversity
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0]
            
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
        
        return current_blocks, uncovered == 0

    # Local search: try to remove blocks that are redundant
    def try_remove(blocks):
        """Try to remove redundant blocks. Returns (new_blocks, success)."""
        n = len(blocks)
        if n == 0:
            return blocks, True
        
        # For each block, check if it's the unique cover for any t-subset
        # If a block is not the unique cover for any t-subset, it can be removed
        # We need to track which t-subsets are covered by which blocks
        
        # Build coverage count: for each t-subset, how many blocks cover it
        # And for each block, which t-subsets does it uniquely cover?
        
        # This is expensive, so we use a different approach:
        # Try removing each block and check if coverage is still complete
        # But that's O(n * C(v,t)) which is too slow
        
        # Alternative: for each block, find t-subsets it covers that are NOT covered by any other block
        # If no such t-subset exists, the block is redundant
        
        # For efficiency, we'll do a simplified version:
        # Count how many blocks cover each t-subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        for block in blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                cover_count[idx] += 1
        
        # A block is redundant if all t-subsets it covers have cover_count > 1
        # But we need to be careful: removing one block might make another redundant
        # So we iterate
        
        changed = True
        new_blocks = list(blocks)
        
        while changed and time.time() < deadline - 2:
            changed = False
            # Recompute cover_count
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for block in new_blocks:
                for ts in itertools.combinations(block, t):
                    idx = t_subset_index[ts]
                    cover_count[idx] += 1
            
            # Find a redundant block
            for i, block in enumerate(new_blocks):
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_index[ts]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                if redundant:
                    new_blocks.pop(i)
                    changed = True
                    break
        
        return new_blocks, len(new_blocks) < len(blocks)

    # Local search: try swapping one block for another
    def try_swap(blocks, max_swaps=100):
        """Try to improve by swapping blocks. Returns (new_blocks, improved)."""
        n = len(blocks)
        if n < 2:
            return blocks, False
        
        improved = False
        new_blocks = list(blocks)
        
        # Compute coverage
        covered = np.zeros(num_t_subsets, dtype=np.int32)
        for block in new_blocks:
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                covered[idx] += 1
        
        # For each block, try to find a replacement that covers all its unique t-subsets
        # and possibly more
        for i in range(n):
            if time.time() > deadline - 5:
                break
            if improved:
                # Recompute coverage after a swap
                covered = np.zeros(num_t_subsets, dtype=np.int32)
                for block in new_blocks:
                    for ts in itertools.combinations(block, t):
                        idx = t_subset_index[ts]
                        covered[idx] += 1
            
            block = new_blocks[i]
            # Find t-subsets uniquely covered by this block
            unique_ts = []
            for ts in itertools.combinations(block, t):
                idx = t_subset_index[ts]
                if covered[idx] == 1:
                    unique_ts.append(ts)
            
            if not unique_ts:
                # Block is redundant, remove it
                new_blocks.pop(i)
                improved = True
                continue
            
            # Try to find a replacement block that covers all unique_ts
            # and has good gain
            best_j = -1
            best_gain = 0
            
            for j in range(num_k_subsets):
                if j in [new_blocks[b] for b in range(len(new_blocks))]:
                    continue
                # Check if j covers all unique_ts
                covers_all = True
                for ts in unique_ts:
                    idx = t_subset_index[ts]
                    if not (masks[j] >> idx) & 1:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                gain = popcount(masks[j] & ~masks[i])
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            
            if best_j >= 0 and best_gain > 0:
                new_blocks[i] = k_subsets[best_j]
                improved = True
        
        return new_blocks, improved

    # Run greedy with random restarts
    best_blocks = None
    best_count = float('inf')
    
    # First, get a greedy solution
    initial_blocks, complete = greedy_init(randomized=False)
    if complete:
        best_blocks = [list(b) for b in initial_blocks]
        best_count = len(initial_blocks)
    
    # Try local search on the initial solution
    if best_blocks is not None:
        improved_blocks, improved = try_remove(best_blocks)
        if improved:
            best_blocks = [list(b) for b in improved_blocks]
            best_count = len(improved_blocks)
    
    # Random restarts
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # Randomized greedy
        blocks, complete = greedy_init(randomized=True)
        if not complete:
            continue
        
        if len(blocks) < best_count:
            best_blocks = [list(b) for b in blocks]
            best_count = len(blocks)
        
        # Try local search
        improved_blocks, improved = try_remove(blocks)
        if improved and len(improved_blocks) < best_count:
            best_blocks = [list(b) for b in improved_blocks]
            best_count = len(improved_blocks)
        
        # Try swaps if we have time
        if time.time() < deadline - 20 and best_count <= 350:
            swapped_blocks, swapped = try_swap(improved_blocks if improved else blocks, max_swaps=50)
            if swapped and len(swapped_blocks) < best_count:
                best_blocks = [list(b) for b in swapped_blocks]
                best_count = len(swapped_blocks)
    
    if best_blocks is None:
        # Fallback
        blocks, _ = greedy_init(randomized=False)
        best_blocks = [list(b) for b in blocks]
    
    return best_blocks
