import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    random.seed(42)
    
    # Precompute comparability
    comp = [[0] * total_masks for _ in range(total_masks)]
    comp_list = [[] for _ in range(total_masks)]
    
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1
                comp_list[i].append(j)
                comp_list[j].append(i)

    # 1. Initialize with a chain
    family = []
    cur = 0
    family.append(cur)
    for i in range(n):
        cur |= (1 << i)
        family.append(cur)
    
    used = set(family)
    
    # 2. Greedily add sets
    for _ in range(m - len(family)):
        best_gain = -1
        best_mask = None
        
        for s in range(total_masks):
            if s in used:
                continue
            
            gain = 0
            for f in family:
                if comp[s][f]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_mask = s
        
        if best_mask is None:
            break
        family.append(best_mask)
        used.add(best_mask)
        
    while len(family) < m:
        s = random.randint(0, total_masks - 1)
        if s not in used:
            family.append(s)
            used.add(s)

    # State variables
    fam = family[:]
    in_fam = [False] * total_masks
    for mask in fam:
        in_fam[mask] = True
        
    sum_comp = [0] * total_masks
    for s in range(total_masks):
        sc = 0
        for x in fam:
            sc += comp[s][x]
        sum_comp[s] = sc
        
    contribs = [sum_comp[fam[i]] for i in range(m)]
    
    current_score = sum(contribs) // 2
    best_score = current_score
    best_fam = fam[:]
    
    deadline = time.time() + 55
    
    T = 5.0
    cooling = 0.999
    T_min = 0.1
    
    no_improve = 0
    top_k = 5 # Number of top candidates to consider for 2-swap

    while time.time() < deadline:
        # 1. Find best 1-swap
        best_1_swap_delta = -1000
        best_1_swap_i = -1
        best_1_swap_s = -1
        
        # Also collect top K candidates for each i for 2-swap
        top_candidates = [[] for _ in range(m)]
        
        for i in range(m):
            old_val = fam[i]
            candidates = []
            for s in range(total_masks):
                if in_fam[s]:
                    continue
                # Gain for 1-swap
                val = sum_comp[s] - comp[s][old_val]
                candidates.append((val, s))
            
            # Sort to get top K
            candidates.sort(reverse=True)
            top_candidates[i] = candidates[:top_k]
            
            if candidates:
                val, s = candidates[0]
                delta = val - contribs[i]
                if delta > best_1_swap_delta:
                    best_1_swap_delta = delta
                    best_1_swap_i = i
                    best_1_swap_s = s

        # 2. Find best 2-swap using top candidates
        best_2_swap_delta = -1000
        best_2_swap_i1 = -1
        best_2_swap_i2 = -1
        best_2_swap_s1 = -1
        best_2_swap_s2 = -1
        
        # Only do 2-swap search if we have time and haven't found a great 1-swap
        # Or always, it's cheap with top_k=5
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                if not top_candidates[i1] or not top_candidates[i2]:
                    continue
                
                old1 = fam[i1]
                old2 = fam[i2]
                
                for delta1, s1 in top_candidates[i1]:
                    # delta1 is gain relative to old1, but contribs[i1] is the current contribution
                    # The formula derived:
                    # Change = Gain1(i1->s1) + Gain1(i2->s2) + comp[old1][old2] - comp[s1][old2] - comp[s2][old1] + comp[s1][s2]
                    # Note: Gain1(i->s) = sum_comp[s] - comp[s][old_val] - contribs[i]
                    # Wait, my previous derivation used Gain1 = sum(comp[s][Ak] - comp[old][Ak]).
                    # In code: val = sum_comp[s] - comp[s][old_val].
                    # contribs[i] = sum_comp[old_val].
                    # So val - contribs[i] is the 1-swap delta.
                    
                    for delta2, s2 in top_candidates[i2]:
                        # Calculate 2-swap delta
                        # d1 = val1 - contribs[i1]
                        # d2 = val2 - contribs[i2]
                        # Interaction term:
                        # old_interaction = comp[old1][old2]
                        # new_interaction = comp[s1][s2]
                        # cross_terms = comp[s1][old2] + comp[s2][old1]
                        # delta_2swap = d1 + d2 + old_interaction - cross_terms + new_interaction
                        
                        d1 = delta1 - contribs[i1]
                        d2 = delta2 - contribs[i2]
                        old_int = comp[old1][old2]
                        new_int = comp[s1][s2]
                        cross = comp[s1][old2] + comp[s2][old1]
                        
                        total_delta = d1 + d2 + old_int - cross + new_int
                        
                        if total_delta > best_2_swap_delta:
                            best_2_swap_delta = total_delta
                            best_2_swap_i1 = i1
                            best_2_swap_i2 = i2
                            best_2_swap_s1 = s1
                            best_2_swap_s2 = s2

        # Decide move
        accept = False
        move_type = 0 # 0: none, 1: 1-swap, 2: 2-swap
        move_i1 = -1
        move_i2 = -1
        move_s1 = -1
        move_s2 = -1
        move_delta = 0
        
        # Compare best 1-swap and best 2-swap
        if best_2_swap_delta > best_1_swap_delta:
            if best_2_swap_delta > 0:
                accept = True
                move_type = 2
                move_i1 = best_2_swap_i1
                move_i2 = best_2_swap_i2
                move_s1 = best_2_swap_s1
                move_s2 = best_2_swap_s2
                move_delta = best_2_swap_delta
            else:
                # Try SA acceptance for 2-swap
                if random.random() < pow(2.718281828, best_2_swap_delta / T):
                    accept = True
                    move_type = 2
                    move_i1 = best_2_swap_i1
                    move_i2 = best_2_swap_i2
                    move_s1 = best_2_swap_s1
                    move_s2 = best_2_swap_s2
                    move_delta = best_2_swap_delta
        else:
            if best_1_swap_delta > 0:
                accept = True
                move_type = 1
                move_i1 = best_1_swap_i
                move_s1 = best_1_swap_s
                move_delta = best_1_swap_delta
            else:
                if random.random() < pow(2.718281828, best_1_swap_delta / T):
                    accept = True
                    move_type = 1
                    move_i1 = best_1_swap_i
                    move_s1 = best_1_swap_s
                    move_delta = best_1_swap_delta

        if accept:
            if move_type == 1:
                old_val = fam[move_i1]
                new_val = move_s1
                
                fam[move_i1] = new_val
                in_fam[old_val] = False
                in_fam[new_val] = True
                
                set_new = set(comp_list[new_val])
                set_old = set(comp_list[old_val])
                to_add = set_new - set_old
                to_rem = set_old - set_new
                
                for s in to_add:
                    sum_comp[s] += 1
                for s in to_rem:
                    sum_comp[s] -= 1
                    
                for k in range(m):
                    if k == move_i1:
                        contribs[k] = sum_comp[new_val]
                    else:
                        contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                        
            elif move_type == 2:
                old1 = fam[move_i1]
                old2 = fam[move_i2]
                new1 = move_s1
                new2 = move_s2
                
                fam[move_i1] = new1
                fam[move_i2] = new2
                
                in_fam[old1] = False
                in_fam[old2] = False
                in_fam[new1] = True
                in_fam[new2] = True
                
                # Update sum_comp
                # Remove old1, old2 contributions
                for s in comp_list[old1]:
                    sum_comp[s] -= 1
                for s in comp_list[old2]:
                    sum_comp[s] -= 1
                
                # Add new1, new2 contributions
                for s in comp_list[new1]:
                    sum_comp[s] += 1
                for s in comp_list[new2]:
                    sum_comp[s] += 1
                    
                # Update contribs
                # For i1:
                contribs[move_i1] = sum_comp[new1]
                # For i2:
                contribs[move_i2] = sum_comp[new2]
                # For others k:
                # contribs[k] changes by (comp[new1][fam[k]] + comp[new2][fam[k]] - comp[old1][fam[k]] - comp[old2][fam[k]])
                # But note fam[move_i1] and fam[move_i2] are already updated.
                # We need to be careful with the update order.
                # It is safer to recalculate contribs for all k if m is small?
                # m=40, so O(m^2) = 1600 ops is fine.
                for k in range(m):
                    if k == move_i1 or k == move_i2:
                        continue
                    contribs[k] += comp[new1][fam[k]] + comp[new2][fam[k]] - comp[old1][fam[k]] - comp[old2][fam[k]]
                    # Note: fam[move_i1] is new1, fam[move_i2] is new2.
                    # So for k != i1, i2, fam[k] is unchanged.
                    # The term comp[new1][fam[k]] is correct.

            current_score = sum(contribs) // 2
            
            if current_score > best_score:
                best_score = current_score
                best_fam = fam[:]
                no_improve = 0
            else:
                no_improve += 1
                
            T *= cooling
            if T < T_min:
                T = T_min
                
            if no_improve > 100:
                fam = best_fam[:]
                in_fam = [False] * total_masks
                for mask in fam:
                    in_fam[mask] = True
                for s in range(total_masks):
                    sc = 0
                    for x in fam:
                        sc += comp[s][x]
                    sum_comp[s] = sc
                contribs = [sum_comp[fam[i]] for i in range(m)]
                current_score = best_score
                T = T_min * 2
                no_improve = 0
        else:
            no_improve += 1
            T *= cooling
            if T < T_min:
                T = T_min

    final_family = best_fam
    
    result = []
    for mask in final_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
