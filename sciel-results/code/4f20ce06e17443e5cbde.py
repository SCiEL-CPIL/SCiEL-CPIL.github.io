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
    
    while time.time() < deadline:
        best_delta = -1000
        best_i = -1
        best_s = -1
        
        for i in range(m):
            old_val = fam[i]
            max_val = -1
            best_s_for_i = -1
            
            for s in range(total_masks):
                if in_fam[s]:
                    continue
                val = sum_comp[s] - comp[s][old_val]
                if val > max_val:
                    max_val = val
                    best_s_for_i = s
            
            if best_s_for_i != -1:
                delta = max_val - contribs[i]
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_s = best_s_for_i
                    
        accept = False
        move_i = -1
        move_s = -1
        move_delta = 0
        
        if best_delta > 0:
            accept = True
            move_i = best_i
            move_s = best_s
            move_delta = best_delta
        else:
            if random.random() < 0.1:
                move_i = random.randint(0, m - 1)
                while True:
                    s = random.randint(0, total_masks - 1)
                    if not in_fam[s]:
                        break
                move_s = s
                
                old_val = fam[move_i]
                new_contrib = sum_comp[move_s] - comp[move_s][old_val]
                move_delta = new_contrib - contribs[move_i]
                
                if move_delta >= 0 or random.random() < pow(2.718281828, move_delta / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            old_val = fam[move_i]
            new_val = move_s
            
            fam[move_i] = new_val
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
                if k == move_i:
                    contribs[k] = sum_comp[new_val]
                else:
                    contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                    
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
