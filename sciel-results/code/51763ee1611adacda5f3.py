import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    points = list(range(v))
    t_subsets = list(itertools.combinations(points, t))
    k_subsets = list(itertools.combinations(points, k))
    
    M = len(t_subsets)
    t_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute masks and t-indices
    block_masks = [0] * len(k_subsets)
    block_t_indices = [[] for _ in range(len(k_subsets))]
    
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_to_idx[t_sub]
            mask |= (1 << idx)
            block_t_indices[i].append(idx)
        block_masks[i] = mask

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple times with different random seeds
    # Estimate time per run: ~5-10s. We can do ~10 runs.
    # But precomputation is shared.
    
    while time.time() < deadline - 15.0: # Leave 15s for final LS and safety
        random.seed() # New random state
        
        uncovered = (1 << M) - 1
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - 15.0:
                break
                
            best_gain = -1
            best_indices = []
            
            for i in range(len(k_subsets)):
                new_covered = block_masks[i] & uncovered
                if new_covered == 0:
                    continue
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_indices = [i]
                elif gain == best_gain:
                    best_indices.append(i)
                    
            if not best_indices:
                break
                
            chosen_idx = random.choice(best_indices)
            selected_indices.append(chosen_idx)
            uncovered &= ~block_masks[chosen_idx]
            
        # Local Search
        if time.time() < deadline - 5.0:
            cover_counts = [0] * M
            for idx in selected_indices:
                for t_idx in block_t_indices[idx]:
                    cover_counts[t_idx] += 1
                    
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                new_selected = []
                for idx in selected_indices:
                    for t_idx in block_t_indices[idx]:
                        cover_counts[t_idx] -= 1
                        
                    redundant = True
                    for t_idx in block_t_indices[idx]:
                        if cover_counts[t_idx] == 0:
                            redundant = False
                            break
                            
                    if redundant:
                        changed = True
                    else:
                        for t_idx in block_t_indices[idx]:
                            cover_counts[t_idx] += 1
                        new_selected.append(idx)
                        
                selected_indices = new_selected
                
        if len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_blocks = [list(k_subsets[i]) for i in selected_indices]
            
    if best_blocks is None:
        # Fallback: return empty or some valid set? 
        # The loop should always produce something if uncovered wasn't empty initially.
        # If v,k,t are such that no block covers anything, it fails. But task implies valid params.
        return []
        
    return best_blocks
