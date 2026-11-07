import time
import random
import math

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
    deadline = start_time + 58.0  # Safety margin
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    
    # Precompute comparability bitsets
    # comp[i] is an integer where bit j is 1 if mask i and mask j are comparable
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j and ((i & j) == i or (i & j) == j):
                ci |= (1 << j)
        comp[i] = ci

    # We fix the empty set (0) and the full set (total_masks - 1)
    # They are comparable with everything.
    # Let's include them in the final result always.
    # We need to choose m - 2 = 38 other sets.
    k = m - 2
    full_mask = total_masks - 1
    
    # Exclude 0 and full_mask from the candidate pool for the variable part
    # Candidates are 1 to full_mask - 1
    candidates = list(range(1, full_mask))
    
    best_score = -1
    best_selection = []
    
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with random k masks from candidates
        selected = random.sample(candidates, k)
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
            
        # Calculate initial score among the k selected
        # Score = sum of comparable pairs within selected
        # We don't need to count pairs with 0 and full_mask as they are constant additions
        # But for comparison, let's just count internal pairs.
        
        def get_internal_score(sel_list, sel_mask):
            score = 0
            for idx, s in enumerate(sel_list):
                # Count comparable in sel_mask, excluding self
                # comp[s] has bits for all comparable.
                # We want bits that are in sel_mask.
                # Since s is in sel_mask, comp[s] & sel_mask includes s? No, comp[s] does not have bit s set (i!=j check).
                # So comp[s] & sel_mask gives comparable elements in the set.
                score += (comp[s] & sel_mask).bit_count()
            return score // 2
        
        current_score = get_internal_score(selected, selected_mask)
        
        temperature = 10.0
        cooling_rate = 0.99
        min_temp = 0.01
        no_improve = 0
        max_no_improve = 5000
        
        while temperature > min_temp and time.time() < deadline:
            improved = False
            
            # 1-Swap Move
            idx = random.randint(0, k - 1)
            old_mask = selected[idx]
            
            # Remove old_mask from selected_mask temporarily for calculation
            sel_without_old = selected_mask & ~(1 << old_mask)
            
            # Loss: comparable pairs between old_mask and others in selected
            loss = (comp[old_mask] & sel_without_old).bit_count()
            
            # Pick a random new_mask from candidates not in selected
            # Optimization: Pre-compute list of candidates not in selected? 
            # Or just sample until we find one.
            new_mask = -1
            attempts = 0
            while new_mask == -1 and attempts < 100:
                cand = random.choice(candidates)
                if not ((selected_mask >> cand) & 1):
                    new_mask = cand
                attempts += 1
            
            if new_mask != -1:
                # Gain: comparable pairs between new_mask and others in selected (excluding old)
                gain = (comp[new_mask] & sel_without_old).bit_count()
                
                delta = gain - loss
                
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                    # Accept
                    selected[idx] = new_mask
                    selected_mask = sel_without_old | (1 << new_mask)
                    current_score += delta
                    improved = True
                    no_improve = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)
                else:
                    no_improve += 1
            
            # 2-Swap Move (occasionally)
            if not improved and random.random() < 0.1:
                idx1 = random.randint(0, k - 1)
                idx2 = random.randint(0, k - 1)
                if idx1 == idx2:
                    continue
                
                old1 = selected[idx1]
                old2 = selected[idx2]
                
                if old1 == old2:
                    continue
                
                sel_without_both = selected_mask & ~(1 << old1) & ~(1 << old2)
                
                # Loss calculation
                # Pairs (old1, z) for z in sel_without_both
                # Pairs (old2, z) for z in sel_without_both
                # Pair (old1, old2)
                loss1 = (comp[old1] & sel_without_both).bit_count()
                loss2 = (comp[old2] & sel_without_both).bit_count()
                if (comp[old1] >> old2) & 1:
                    total_loss = loss1 + loss2 + 1
                else:
                    total_loss = loss1 + loss2
                
                # Pick two new masks
                new1 = -1
                new2 = -1
                attempts = 0
                while (new1 == -1 or new2 == -1) and attempts < 200:
                    cand = random.choice(candidates)
                    if not ((selected_mask >> cand) & 1) and cand != old1 and cand != old2:
                        if new1 == -1:
                            new1 = cand
                        elif cand != new1:
                            new2 = cand
                    attempts += 1
                
                if new1 != -1 and new2 != -1:
                    gain1 = (comp[new1] & sel_without_both).bit_count()
                    gain2 = (comp[new2] & sel_without_both).bit_count()
                    if (comp[new1] >> new2) & 1:
                        total_gain = gain1 + gain2 + 1
                    else:
                        total_gain = gain1 + gain2
                    
                    delta = total_gain - total_loss
                    
                    if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                        selected[idx1] = new1
                        selected[idx2] = new2
                        selected_mask = sel_without_both | (1 << new1) | (1 << new2)
                        current_score += delta
                        improved = True
                        no_improve = 0
                        
                        if current_score > best_score:
                            best_score = current_score
                            best_selection = list(selected)
                    else:
                        no_improve += 1
            
            if improved:
                pass
            else:
                temperature *= cooling_rate
                
            if no_improve > max_no_improve:
                break

    # Final Polish: Hill Climbing on best_selection
    if best_selection and time.time() < deadline:
        selected = list(best_selection)
        selected_mask = 0
        for s in selected:
            selected_mask |= (1 << s)
        current_score = best_score
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(k):
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                # Scan all candidates for best swap
                # This is O(N_candidates) per step, which is ~4000.
                # 38 steps * 4000 = 152k operations per pass.
                # Very fast.
                for cand in candidates:
                    if (selected_mask >> cand) & 1:
                        continue
                    gain = (comp[cand] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = cand
                
                if best_new != -1 and best_delta > 0:
                    selected[idx] = best_new
                    selected_mask = sel_without_old | (1 << best_new)
                    current_score += best_delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)

    # Construct final result
    # Add 0 and full_mask
    final_masks = [0, full_mask] + best_selection
    
    # Convert to lists of integers
    result = []
    for mask in final_masks:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    # Ensure types are correct
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
        
    return final_result
