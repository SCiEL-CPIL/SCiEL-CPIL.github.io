import time
import random

def construct():
    n = 12
    m = 40
    deadline = time.time() + 85

    def is_comparable(a, b):
        if a == b:
            return False
        return (a & ~b) == 0 or (b & ~a) == 0

    def count_comparable_pairs(config):
        cnt = 0
        for i in range(len(config)):
            for j in range(i + 1, len(config)):
                if is_comparable(config[i], config[j]):
                    cnt += 1
        return cnt

    k_init = 7
    init_pool = list(range(1 << k_init))
    
    selected = []
    available_init = set(init_pool)
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in available_init:
            score = 0
            for s in selected:
                if is_comparable(mask, s):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
        available_init.remove(best_mask)
        
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    selected_set = set(selected)

    small_pool = list(range(1 << 7))
    
    sp_size = len(small_pool)
    sp_comp = [[0] * sp_size for _ in range(sp_size)]
    for i in range(sp_size):
        for j in range(i + 1, sp_size):
            if is_comparable(small_pool[i], small_pool[j]):
                sp_comp[i][j] = 1
                sp_comp[j][i] = 1
    
    mask_to_idx = {sm: i for i, sm in enumerate(small_pool)}

    def get_comp_count(mask, config, skip_idx=None):
        cnt = 0
        mask_idx = mask_to_idx.get(mask, -1)
        for j, s in enumerate(config):
            if j == skip_idx:
                continue
            if mask_idx >= 0:
                s_idx = mask_to_idx.get(s, -1)
                if s_idx >= 0:
                    if sp_comp[mask_idx][s_idx]:
                        cnt += 1
                else:
                    if is_comparable(mask, s):
                        cnt += 1
            else:
                if is_comparable(mask, s):
                    cnt += 1
        return cnt

    while time.time() < deadline:
        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            old_mask = selected[i]
            loss = get_comp_count(old_mask, selected, i)
            
            best_new_mask = -1
            best_delta = 0
            
            for new_mask in small_pool:
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
        
        if not improved:
            for extra_bit in range(7, 12):
                if time.time() > deadline:
                    break
                bit_val = 1 << extra_bit
                ext_pool = [sm | bit_val for sm in small_pool]
                
                for i in range(m):
                    if time.time() > deadline:
                        break
                    old_mask = selected[i]
                    loss = get_comp_count(old_mask, selected, i)
                    
                    best_new_mask = -1
                    best_delta = 0
                    
                    for new_mask in ext_pool:
                        if new_mask in selected_set:
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
            
            if not improved:
                for _ in range(200):
                    if time.time() > deadline:
                        break
                    i1 = random.randint(0, m - 1)
                    i2 = random.randint(0, m - 1)
                    if i1 == i2:
                        continue
                    
                    m1 = random.choice(small_pool)
                    m2 = random.choice(small_pool)
                    
                    if m1 == m2:
                        continue
                    if m1 in selected_set and m1 != selected[i1]:
                        continue
                    if m2 in selected_set and m2 != selected[i2]:
                        continue
                    
                    old_m1 = selected[i1]
                    old_m2 = selected[i2]
                    
                    selected[i1] = m1
                    selected[i2] = m2
                    selected_set.remove(old_m1)
                    selected_set.remove(old_m2)
                    selected_set.add(m1)
                    selected_set.add(m2)
                    
                    new_score = count_comparable_pairs(selected)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        selected[i1] = old_m1
                        selected[i2] = old_m2
                        selected_set.remove(m1)
                        selected_set.remove(m2)
                        selected_set.add(old_m1)
                        selected_set.add(old_m2)
                
                if not improved:
                    break

        if current_score > best_score:
            best_score = current_score
            best_config = selected[:]

    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
