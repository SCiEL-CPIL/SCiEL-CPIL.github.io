import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    masks = [0] * num_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << t_subset_index[t_sub])
        masks[i] = mask
        
    full_mask = (1 << num_t_subsets) - 1
    
    best_solution = None
    best_count = float('inf')
    
    def run_greedy_reduction(seed):
        random.seed(seed)
        uncovered = full_mask
        blocks = []
        
        # Greedy Phase
        while uncovered:
            best_gain = 0
            best_candidates = []
            
            for i in range(num_k_subsets):
                gain = (masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                break
                
            chosen_idx = random.choice(best_candidates)
            blocks.append(chosen_idx)
            uncovered &= ~masks[chosen_idx]
            
        # Reduction Phase
        counts = [0] * num_t_subsets
        for b_idx in blocks:
            m = masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
                
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                removable = True
                m = masks[b_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] == 1:
                        removable = False
                        break
                    m ^= lsb
                if removable:
                    m = masks[b_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
        return blocks

    seed = 0
    while True:
        current_time = time.time()
        if current_time > deadline:
            break
            
        res = run_greedy_reduction(seed)
        
        if len(res) < best_count:
            best_count = len(res)
            best_solution = res
            
        seed += 1
        
        if time.time() > deadline - 2.0:
            break

    if best_solution is None:
        return []

    final_blocks = [list(k_subsets[i]) for i in best_solution]
    return final_blocks
