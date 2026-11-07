import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85

    # Precompute comparability for all 2^12 subsets
    total_masks = 1 << n
    # comp[i][j] = 1 if mask i and mask j are comparable (one is strict subset of other)
    # We'll use a flat structure for speed
    # Actually, let's precompute for a working pool of size 2^8 = 256 for speed
    # But we need to allow expansion. Let's use 2^9 = 512.
    
    pool_size = 512  # 2^9
    pool = list(range(pool_size))
    
    # Precompute comparability matrix for the pool
    comp = [[0] * pool_size for _ in range(pool_size)]
    for i in range(pool_size):
        mi = pool[i]
        for j in range(i + 1, pool_size):
            mj = pool[j]
            if mi == mj:
                continue
            if (mi & ~mj) == 0:  # mi is subset of mj
                comp[i][j] = 1
                comp[j][i] = 1
            elif (mj & ~mi) == 0:  # mj is subset of mi
                comp[i][j] = 1
                comp[j][i] = 1
    
    mask_to_idx = {sm: i for i, sm in enumerate(pool)}

    def get_comp_count(mask, config, skip_idx=None):
        """Count how many sets in config are comparable with mask."""
        cnt = 0
        mask_idx = mask_to_idx.get(mask, -1)
        for j, s in enumerate(config):
            if j == skip_idx:
                continue
            if mask_idx >= 0:
                s_idx = mask_to_idx.get(s, -1)
                if s_idx >= 0:
                    if comp[mask_idx][s_idx]:
                        cnt += 1
                else:
                    # s is not in pool, compute directly
                    if mask == s:
                        continue
                    if (mask & ~s) == 0 or (s & ~mask) == 0:
                        cnt += 1
            else:
                if mask == s:
                    continue
                if (mask & ~s) == 0 or (s & ~mask) == 0:
                    cnt += 1
        return cnt

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                a, b = config[i], config[j]
                if a == b:
                    continue
                if (a & ~b) == 0 or (b & ~a) == 0:
                    cnt += 1
        return cnt

    # Greedy initialization: start with a long chain, then add sets greedily
    # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,...,11}
    chain = [sum(1 << k for k in range(i)) for i in range(n + 1)]  # 13 sets
    selected = chain[:]
    selected_set = set(selected)
    
    # Greedily add remaining sets
    for _ in range(m - len(selected)):
        best_mask = -1
        best_score = -1
        for mask in pool:
            if mask in selected_set:
                continue
            score = get_comp_count(mask, selected)
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            # Try outside pool
            for mask in range(total_masks):
                if mask in selected_set:
                    continue
                score = get_comp_count(mask, selected)
                if score > best_score:
                    best_score = score
                    best_mask = mask
        selected.append(best_mask)
        selected_set.add(best_mask)
    
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    
    # Simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    rng = random.Random(42)
    
    while time.time() < deadline and T > T_min:
        i = rng.randint(0, m - 1)
        old_mask = selected[i]
        loss = get_comp_count(old_mask, selected, i)
        
        # Pick a new mask
        if rng.random() < 0.7:
            new_mask = rng.choice(pool)
        else:
            new_mask = rng.randint(0, total_masks - 1)
        
        if new_mask == old_mask:
            continue
        if new_mask in selected_set:
            continue
        
        gain = get_comp_count(new_mask, selected, i)
        delta = gain - loss
        
        if delta >= 0 or rng.random() < pow(2.718281828, delta / T):
            current_score += delta
            selected[i] = new_mask
            selected_set.remove(old_mask)
            selected_set.add(new_mask)
            
            if current_score > best_score:
                best_score = current_score
                best_config = selected[:]
        
        T *= cooling
    
    # Final local search pass
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(m):
            old_mask = selected[i]
            loss = get_comp_count(old_mask, selected, i)
            best_new_mask = -1
            best_delta = 0
            for new_mask in pool:
                if new_mask == old_mask or new_mask in selected_set:
                    continue
                gain = get_comp_count(new_mask, selected, i)
                delta = gain - loss
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            if best_new_mask != -1:
                current_score += best_delta
                selected[i] = best_new_mask
                selected_set.remove(old_mask)
                selected_set.add(best_new_mask)
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_config = selected[:]
    
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    
    return result
