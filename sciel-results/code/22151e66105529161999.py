import itertools
import time
import random

_PRECOMP = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _PRECOMP:
        return _PRECOMP[key]
        
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    idx_to_subset = [s for s in all_t_subsets] # List for index access
    full_mask = (1 << len(all_t_subsets)) - 1
    
    partial_masks = {}
    for i in range(1, k + 1):
        for comb in itertools.combinations(range(v), i):
            mask = 0
            if i >= t:
                for t_sub in itertools.combinations(comb, t):
                    idx = subset_to_idx[t_sub]
                    mask |= (1 << idx)
            partial_masks[comb] = mask
            
    data = {
        'full_mask': full_mask,
        'partial_masks': partial_masks,
        'idx_to_subset': idx_to_subset
    }
    _PRECOMP[key] = data
    return data

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0
    
    data = _get_precomputed(v, k, t)
    full_mask = data['full_mask']
    partial_masks = data['partial_masks']
    idx_to_subset = data['idx_to_subset']
    
    best_blocks = None
    best_count = float('inf')
    
    def get_set_bit_indices(mask):
        indices = []
        i = 0
        while mask:
            if mask & 1:
                indices.append(i)
            mask >>= 1
            i += 1
        return indices

    def greedy_pass(seed_val):
        rng = random.Random(seed_val)
        uncovered = full_mask
        blocks = []
        
        while uncovered:
            indices = get_set_bit_indices(uncovered)
            if not indices:
                break
                
            start_idx = rng.choice(indices)
            start_subset = idx_to_subset[start_idx]
            block = list(start_subset)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                # No need to shuffle candidates if we collect all max gain
                # and pick random from them.
                
                max_gain = -1
                max_gain_cands = []
                
                for x in candidates:
                    new_block_tuple = tuple(sorted(block + [x]))
                    mask = partial_masks[new_block_tuple]
                    gain = (uncovered & mask).bit_count()
                    
                    if gain > max_gain:
                        max_gain = gain
                        max_gain_cands = [x]
                    elif gain == max_gain:
                        max_gain_cands.append(x)
                        
                if max_gain_cands:
                    best_cand = rng.choice(max_gain_cands)
                else:
                    best_cand = candidates[0]
                    
                block.append(best_cand)
                
            final_block_tuple = tuple(sorted(block))
            block_mask = partial_masks[final_block_tuple]
            uncovered &= ~block_mask
            blocks.append(final_block_tuple)
            
        return blocks

    seed = 0
    # Run passes until deadline
    while time.time() < deadline:
        # Run a batch of passes
        for _ in range(100):
            if time.time() > deadline:
                break
            res = greedy_pass(seed)
            if len(res) < best_count:
                best_count = len(res)
                best_blocks = res
            seed += 1
        
        # If we've run many seeds, maybe increase batch size or just continue
        # 100 passes is fast.
        pass

    if best_blocks is None:
        res = greedy_pass(0)
        best_blocks = res

    return [list(b) for b in best_blocks]
