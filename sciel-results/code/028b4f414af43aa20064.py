import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin before 120s
    
    # Precomputation
    # 1. Enumerate all t-subsets and assign indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Enumerate all k-subsets and compute their coverage bitmask
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask where bit j is 1 if k_subset i covers t_subset j
    masks = [0] * num_k_subsets
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        masks[i] = mask
        
    # Helper to count bits
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple random restarts
    num_restarts = 50000  # We will limit by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        while uncovered:
            if time.time() > deadline:
                break
                
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
                
            # Pick a random candidate to add diversity
            chosen_idx = random.choice(candidates)
            current_blocks.append(k_subsets[chosen_idx])
            used_blocks.add(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
        if uncovered == 0 and len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = [list(b) for b in current_blocks]

    if best_blocks is None:
        # Fallback to simple greedy if randomization failed (unlikely)
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
            current_blocks.append(k_subsets[best_block_idx])
            used_blocks.add(best_block_idx)
            uncovered &= ~masks[best_block_idx]
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
