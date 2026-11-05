import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all t-subsets and map to indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)

    # 3. Precompute for each k-subset:
    #    - List of t-subset indices it covers (k_t_idxs)
    #    - Bitmask of t-subset indices it covers (k_t_masks)
    k_t_idxs = []
    k_t_masks = []
    
    for k_sub in all_k_subs:
        t_idxs = []
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            t_idxs.append(idx)
            mask |= (1 << idx)
        k_t_idxs.append(t_idxs)
        k_t_masks.append(mask)

    # 4. Greedy Set Cover using Bitmasks
    def greedy_cover_bitmask(seed):
        rng = random.Random(seed)
        # Initial uncovered mask: all bits set
        uncovered_mask = (1 << n_t) - 1
        
        selected = []
        
        while uncovered_mask != 0:
            best_idx = -1
            best_gain = -1
            
            for i in range(n_k):
                gain_mask = uncovered_mask & k_t_masks[i]
                if gain_mask:
                    gain = gain_mask.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                    elif gain == best_gain:
                        # Random tie breaking
                        if rng.random() < 0.5:
                            best_idx = i
            
            if best_idx == -1:
                break
                
            selected.append(best_idx)
            uncovered_mask &= ~k_t_masks[best_idx]
            
        return selected

    def reduce_blocks(blocks_idx):
        if not blocks_idx:
            return blocks_idx
            
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks_idx:
            for t_idx in k_t_idxs[b_idx]:
                cover_count[t_idx] += 1
                
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks_idx:
                is_essential = False
                for t_idx in k_t_idxs[b_idx]:
                    if cover_count[t_idx] == 1:
                        is_essential = True
                        break
                if is_essential:
                    new_blocks.append(b_idx)
                else:
                    changed = True
                    for t_idx in k_t_idxs[b_idx]:
                        cover_count[t_idx] -= 1
            blocks_idx = new_blocks
        return blocks_idx

    # Run greedy
    best_blocks = greedy_cover_bitmask(0)
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    # Try a few more seeds if time permits
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        for seed in [1, 2, 3, 42, 123]:
            if time.time() >= deadline - 5.0:
                break
            b_idx = greedy_cover_bitmask(seed)
            b_idx = reduce_blocks(b_idx)
            if b_idx and len(b_idx) < best_len:
                best_blocks = b_idx
                best_len = len(b_idx)
                
    return [list(all_k_subs[b_idx]) for b_idx in best_blocks]
