import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85

    total_masks = 1 << n
    
    # Precompute comparability for all 4096 subsets
    # comp[i][j] = 1 if mask i and mask j are comparable (one is strict subset of other)
    comp = [[0] * total_masks for _ in range(total_masks)]
    for i in range(total_masks):
        mi = i
        for j in range(i + 1, total_masks):
            mj = j
            if (mi & ~mj) == 0:  # mi is subset of mj
                comp[i][j] = 1
                comp[j][i] = 1
            elif (mj & ~mi) == 0:  # mj is subset of mi
                comp[i][j] = 1
                comp[j][i] = 1
    
    def get_comp_count(mask, config, skip_idx=None):
        """Count how many sets in config are comparable with mask."""
        cnt = 0
        for j, s in enumerate(config):
            if j == skip_idx:
                continue
            if comp[mask][s]:
                cnt += 1
        return cnt

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if comp[config[i]][config[j]]:
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
            for new_mask in range(total_masks):
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
