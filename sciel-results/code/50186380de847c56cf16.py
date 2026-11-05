import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Pre-computation
    points = range(v)
    k_subsets = list(itertools.combinations(points, k))
    n_k = len(k_subsets)
    
    t_subsets = list(itertools.combinations(points, t))
    t_map = {ts: i for i, ts in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    masks = [0] * n_k
    for i, ksub in enumerate(k_subsets):
        mask = 0
        for tsub in itertools.combinations(ksub, t):
            mask |= (1 << t_map[tsub])
        masks[i] = mask
        
    full_mask = (1 << n_t) - 1
    
    best_blocks = None
    best_len = float('inf')
    
    seed = 0
    while time.time() < deadline:
        rng = random.Random(seed)
        seed += 1
        
        uncovered = full_mask
        block_indices = []
        
        # Greedy Phase
        while uncovered:
            best_gain = -1
            best_idx = -1
            for i in range(n_k):
                gain = (uncovered & masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                elif gain == best_gain and gain > 0:
                    if rng.random() < 0.5:
                        best_idx = i
            
            if best_idx == -1 or best_gain == 0:
                break
                
            block_indices.append(best_idx)
            uncovered &= ~masks[best_idx]
            
        # Redundancy Removal Phase
        active = list(block_indices)
        changed = True
        while changed and active:
            changed = False
            U = 0
            for idx in active:
                U |= masks[idx]
                
            new_active = []
            for idx in active:
                if (U & ~masks[idx]) == full_mask:
                    changed = True
                else:
                    new_active.append(idx)
                    
            active = new_active
            
        if len(active) < best_len:
            best_len = len(active)
            best_blocks = [k_subsets[i] for i in active]
            
    return best_blocks
