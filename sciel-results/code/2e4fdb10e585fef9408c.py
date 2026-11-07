import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 80  # Safety margin before 90s kill

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

    # Greedy Initialization
    selected = []
    available = set(range(total_masks))
    
    for _ in range(m):
        best_mask = -1
        best_score = -1
        for mask in available:
            score = 0
            for s in selected:
                if is_comparable(mask, s):
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        selected.append(best_mask)
        available.remove(best_mask)
        
    current_score = count_comparable_pairs(selected)
    best_config = selected[:]
    best_score = current_score
    
    # Local Search with compatibility count array for O(1) swap evaluation
    selected_set = set(selected)
    
    # Precompute compatibility: compat[mask] = number of selected sets comparable to mask
    # This is O(total_masks * m) to build, O(1) per swap evaluation
    def build_compat(selected_arr, selected_set_local):
        compat = [0] * total_masks
        for mask in range(total_masks):
            c = 0
            for s in selected_arr:
                if is_comparable(mask, s):
                    c += 1
            compat[mask] = c
        return compat
    
    compat = build_compat(selected, selected_set)
    
    no_improve_passes = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        improved = False
        # Iterate through indices multiple times per "pass" for steepest descent
        for i in range(m):
            old_mask = selected[i]
            # loss = number of comparable pairs old_mask has with other selected sets
            # compat[old_mask] counts comparability with ALL selected including itself? No, is_comparable(x,x)=False
            # So compat[old_mask] = number of other selected sets comparable to old_mask = loss
            loss = compat[old_mask]
            
            best_new_mask = -1
            best_delta = 0
            
            for new_mask in range(total_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in selected_set:
                    continue
                
                # gain = compat[new_mask] (comparability with all current selected)
                # delta = gain - loss
                delta = compat[new_mask] - loss
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask != -1:
                # Perform swap
                selected_set.remove(old_mask)
                selected_set.add(best_new_mask)
                selected[i] = best_new_mask
                current_score += best_delta
                
                # Update compat array:
                # For all masks, adjust by:
                # -1 if comparable to old_mask (removed)
                # +1 if comparable to best_new_mask (added)
                for mask in range(total_masks):
                    if is_comparable(mask, old_mask):
                        compat[mask] -= 1
                    if is_comparable(mask, best_new_mask):
                        compat[mask] += 1
                
                improved = True
        
        if current_score > best_score:
            best_score = current_score
            best_config = selected[:]
        
        if improved:
            no_improve_passes = 0
        else:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                # Perturbation: swap 2 random elements with random masks
                for _attempt in range(20):
                    i1 = random.randint(0, m - 1)
                    i2 = random.randint(0, m - 1)
                    if i1 == i2:
                        continue
                    m1 = random.randint(0, total_masks - 1)
                    m2 = random.randint(0, total_masks - 1)
                    if m1 in selected_set or m2 in selected_set:
                        continue
                    if m1 == m2:
                        continue
                    if m1 == selected[i1] or m2 == selected[i2]:
                        continue
                    
                    # Compute delta for both swaps
                    # Remove selected[i1], add m1; remove selected[i2], add m2
                    old1 = selected[i1]
                    old2 = selected[i2]
                    
                    # loss1 = compat[old1] but need to be careful about double-counting
                    # Let's just recompute score after swap to verify
                    old_score = current_score
                    selected_set.remove(old1)
                    selected_set.remove(old2)
                    selected_set.add(m1)
                    selected_set.add(m2)
                    selected[i1] = m1
                    selected[i2] = m2
                    new_score = count_comparable_pairs(selected)
                    
                    if new_score > old_score:
                        current_score = new_score
                        # Rebuild compat
                        compat = build_compat(selected, selected_set)
                        no_improve_passes = 0
                        break
                    else:
                        # Revert
                        selected_set.remove(m1)
                        selected_set.remove(m2)
                        selected_set.add(old1)
                        selected_set.add(old2)
                        selected[i1] = old1
                        selected[i2] = old2
                        current_score = old_score
                else:
                    # No successful perturbation, break
                    break
    
    # Convert to lists of ints
    result = []
    for mask in best_config:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
        
    return result
