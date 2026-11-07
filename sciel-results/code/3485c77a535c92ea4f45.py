# EVOLVE-BLOCK-START
import time
import random

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            sj = sets[j]
            if si and sj:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = set(si)
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
            elif si and not sj:
                cnt += 1
            elif not si and sj:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comparability bitsets
    comp_bitset = [0] * total_masks
    
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp_bitset[i] = ci
    
    # Greedy Construction with density consideration
    selected = []
    selected_mask = 0
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for k in range(total_masks):
            if (selected_mask >> k) & 1:
                continue
            score = (comp_bitset[k] & selected_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = k
        
        selected.append(best_mask)
        selected_mask |= (1 << best_mask)
    
    # Calculate initial score
    current_score = 0
    for i in range(m):
        current_score += (comp_bitset[selected[i]] & selected_mask).bit_count()
    current_score //= 2
    
    best_score = current_score
    best_selected = list(selected)
    
    # Local Search with simulated annealing
    improved = True
    temperature = 5.0
    cooling_rate = 0.95
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try single swaps
        for idx in range(m):
            if time.time() >= deadline:
                break
            
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
            
            # Try adding all non-selected masks
            for new_mask in range(total_masks):
                if (selected_mask >> new_mask) & 1:
                    continue
                
                gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                if delta > 0:
                    # Accept improving move
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = list(selected)
                    
                    improved = True
                    break
                elif delta < 0 and random.random() < pow(2.718281828, delta / max(temperature, 0.1)):
                    # Accept worsening move with probability based on temperature
                    selected[idx] = new_mask
                    selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                    current_score += delta
                    improved = True
                    break
        
        # Try 2-swap moves (remove two, add two)
        if time.time() < deadline:
            for idx1 in range(m):
                if time.time() >= deadline:
                    break
                for idx2 in range(idx1 + 1, m):
                    if time.time() >= deadline:
                        break
                    
                    old_mask1 = selected[idx1]
                    old_mask2 = selected[idx2]
                    sel_without_both = selected_mask & ~(1 << old_mask1) & ~(1 << old_mask2)
                    
                    loss1 = (comp_bitset[old_mask1] & sel_without_both).bit_count()
                    loss2 = (comp_bitset[old_mask2] & sel_without_both).bit_count()
                    total_loss = loss1 + loss2 + (1 if (comp_bitset[old_mask1] >> old_mask2) & 1 else 0)
                    
                    # Try pairs of non-selected masks
                    found = False
                    for new_mask1 in range(total_masks):
                        if found or (selected_mask >> new_mask1) & 1:
                            continue
                        for new_mask2 in range(total_masks):
                            if found or new_mask2 == new_mask1 or (selected_mask >> new_mask2) & 1:
                                continue
                            
                            gain1 = (comp_bitset[new_mask1] & sel_without_both).bit_count()
                            gain2 = (comp_bitset[new_mask2] & sel_without_both).bit_count()
                            total_gain = gain1 + gain2 + (1 if (comp_bitset[new_mask1] >> new_mask2) & 1 else 0)
                            
                            delta = total_gain - total_loss
                            
                            if delta > 0:
                                selected[idx1] = new_mask1
                                selected[idx2] = new_mask2
                                selected_mask = (selected_mask & ~(1 << old_mask1) & ~(1 << old_mask2)) | (1 << new_mask1) | (1 << new_mask2)
                                current_score += delta
                                
                                if current_score > best_score:
                                    best_score = current_score
                                    best_selected = list(selected)
                                
                                improved = True
                                found = True
                                break
                            elif delta < 0 and random.random() < pow(2.718281828, delta / max(temperature, 0.1)):
                                selected[idx1] = new_mask1
                                selected[idx2] = new_mask2
                                selected_mask = (selected_mask & ~(1 << old_mask1) & ~(1 << old_mask2)) | (1 << new_mask1) | (1 << new_mask2)
                                current_score += delta
                                improved = True
                                found = True
                                break
                    if found:
                        break
        
        temperature *= cooling_rate
    
    # Final polish: try more aggressive local search
    if time.time() < deadline - 5:
        for _ in range(100):
            if time.time() >= deadline - 5:
                break
            
            improved_polish = False
            for idx in range(m):
                if time.time() >= deadline - 5:
                    break
                
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp_bitset[old_mask] & sel_without_old).bit_count()
                
                for new_mask in range(total_masks):
                    if (selected_mask >> new_mask) & 1:
                        continue
                    
                    gain = (comp_bitset[new_mask] & sel_without_old).bit_count()
                    delta = gain - loss
                    
                    if delta > 0:
                        selected[idx] = new_mask
                        selected_mask = (selected_mask & ~(1 << old_mask)) | (1 << new_mask)
                        current_score += delta
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_selected = list(selected)
                        
                        improved_polish = True
                        break
            
            if not improved_polish:
                break
    
    # Convert best_selected to lists of integers
    result = []
    for mask in best_selected:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
        
    return final_result
# EVOLVE-BLOCK-END
