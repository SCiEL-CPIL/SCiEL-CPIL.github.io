import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    # Precompute masks for all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    cover_masks = [0] * num_k_subsets
    for i, block in enumerate(k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            mask |= (1 << t_subset_index[t_sub])
        cover_masks[i] = mask
        
    def greedy_construct(randomize=False):
        uncovered = (1 << num_t_subsets) - 1
        blocks = []
        used = [False] * num_k_subsets
        
        while uncovered:
            best_gain = 0
            best_blocks = []
            for i in range(num_k_subsets):
                if used[i]:
                    continue
                gain = (uncovered & cover_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [i]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(i)
            
            if not best_blocks:
                break
                
            if randomize and len(best_blocks) > 1:
                chosen = random.choice(best_blocks)
            else:
                chosen = best_blocks[0]
                
            blocks.append(chosen)
            used[chosen] = True
            uncovered &= ~cover_masks[chosen]
            
            if time.time() > deadline:
                return None # Indicate timeout/incomplete
                
        return blocks

    def local_search(blocks):
        cover_count = [0] * num_t_subsets
        for b_idx in blocks:
            mask = cover_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        current_blocks = blocks[:]
        
        while improved:
            improved = False
            random.shuffle(current_blocks)
            for i in range(len(current_blocks) - 1, -1, -1):
                b_idx = current_blocks[i]
                mask = cover_masks[b_idx]
                
                is_redundant = True
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                    m ^= lsb
                    
                if is_redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        m ^= lsb
                    current_blocks.pop(i)
                    improved = True
            
            if time.time() > deadline:
                break
                
        return current_blocks

    def perturb_and_repair(blocks):
        if not blocks:
            return blocks
            
        rem_idx = random.randint(0, len(blocks) - 1)
        remaining_blocks = [b for i, b in enumerate(blocks) if i != rem_idx]
        
        cover_count = [0] * num_t_subsets
        for b_idx in remaining_blocks:
            mask = cover_masks[b_idx]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        uncovered = 0
        for i in range(num_t_subsets):
            if cover_count[i] == 0:
                uncovered |= (1 << i)
                
        if not uncovered:
            return remaining_blocks
            
        new_blocks = remaining_blocks[:]
        used = set(remaining_blocks)
        
        while uncovered:
            best_gain = 0
            best_blocks = []
            for i in range(num_k_subsets):
                if i in used:
                    continue
                gain = (uncovered & cover_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_blocks = [i]
                elif gain == best_gain and gain > 0:
                    best_blocks.append(i)
                    
            if not best_blocks:
                break
                
            chosen = random.choice(best_blocks)
            new_blocks.append(chosen)
            used.add(chosen)
            uncovered &= ~cover_masks[chosen]
            
            if time.time() > deadline:
                return None
                
        return new_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Initial Greedy
    init_blocks = greedy_construct(randomize=False)
    if init_blocks:
        init_blocks = local_search(init_blocks)
        best_blocks = init_blocks
        best_len = len(init_blocks)
        
    # Randomized Greedy + Local Search
    while time.time() < deadline:
        rand_blocks = greedy_construct(randomize=True)
        if rand_blocks:
            rand_blocks = local_search(rand_blocks)
            if len(rand_blocks) < best_len:
                best_blocks = rand_blocks
                best_len = len(rand_blocks)
                
        # Perturbation
        if best_blocks and time.time() < deadline:
            perturbed = perturb_and_repair(best_blocks)
            if perturbed:
                perturbed = local_search(perturbed)
                if len(perturbed) < best_len:
                    best_blocks = perturbed
                    best_len = len(perturbed)

    if not best_blocks:
        # Fallback
        uncovered = (1 << num_t_subsets) - 1
        blocks = []
        while uncovered:
            best_gain = 0
            best_idx = -1
            for i in range(num_k_subsets):
                gain = (uncovered & cover_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            blocks.append(best_idx)
            uncovered &= ~cover_masks[best_idx]
        best_blocks = blocks

    return [list(k_subsets[i]) for i in best_blocks]
