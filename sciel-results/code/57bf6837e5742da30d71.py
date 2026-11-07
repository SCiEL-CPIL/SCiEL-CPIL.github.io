import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Precompute comparability? 
    # 4096 * 4096 is 16M bools. Might be memory heavy or slow to build.
    # On-the-fly check is (a & b) == a. Fast.
    
    def is_comp(a, b):
        return (a & b) == a or (a & b) == b

    def calc_score(family):
        score = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                if is_comp(family[i], family[j]):
                    score += 1
        return score

    def greedy_init():
        family = []
        in_family = [False] * total_masks
        for _ in range(m):
            best_mask = -1
            best_count = -1
            # To avoid always picking the same first set, we can start with a random one or 0
            if not family:
                # Start with 0 (empty set) is good as it's subset of everything
                best_mask = 0 
                # Or random
                # best_mask = random.choice(all_masks)
            else:
                for mask in all_masks:
                    if in_family[mask]:
                        continue
                    count = 0
                    for s in family:
                        if is_comp(mask, s):
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_mask = mask
            family.append(best_mask)
            in_family[best_mask] = True
        return family

    def local_search(family, time_limit):
        in_family = [False] * total_masks
        for mask in family:
            in_family[mask] = True
            
        current_score = calc_score(family)
        start_time = time.time()
        
        # Precompute degrees? No, update dynamically or recalc if simple.
        # Recalculating score for a swap:
        # Remove S_i, Add S_j.
        # Delta = (comp(S_j, others) - comp(S_i, others))
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            for i in range(m):
                s_i = family[i]
                # Calculate current contribution of s_i
                deg_i = 0
                for j in range(m):
                    if i != j and is_comp(s_i, family[j]):
                        deg_i += 1
                
                best_j = -1
                best_deg = -1
                best_mask = -1
                
                # Search for best replacement
                for mask in all_masks:
                    if in_family[mask]:
                        continue
                    deg_j = 0
                    for k in range(m):
                        if i != k and is_comp(mask, family[k]):
                            deg_j += 1
                    if deg_j > best_deg:
                        best_deg = deg_j
                        best_j = mask # Actually we just need the mask
                        best_mask = mask

                if best_mask != -1 and best_deg > deg_i:
                    # Swap
                    in_family[s_i] = False
                    in_family[best_mask] = True
                    family[i] = best_mask
                    current_score += best_deg - deg_i
                    improved = True
        
        return family, current_score

    start_time = time.time()
    deadline = start_time + 50.0 # 50s limit to be safe
    
    # 1. Greedy Initialization
    family = greedy_init()
    
    # 2. Local Search
    # We can run multiple rounds if time permits
    # But local_search above runs until no improvement or time limit.
    # Let's add a timeout check inside the while loop of local_search properly.
    
    # Refine local_search to respect deadline strictly
    def local_search_deadline(family, deadline):
        in_family = [False] * total_masks
        for mask in family:
            in_family[mask] = True
            
        current_score = calc_score(family)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(m):
                if time.time() >= deadline:
                    break
                s_i = family[i]
                deg_i = 0
                for j in range(m):
                    if i != j and is_comp(s_i, family[j]):
                        deg_i += 1
                
                best_deg = -1
                best_mask = -1
                
                for mask in all_masks:
                    if in_family[mask]:
                        continue
                    deg_j = 0
                    for k in range(m):
                        if i != k and is_comp(mask, family[k]):
                            deg_j += 1
                    if deg_j > best_deg:
                        best_deg = deg_j
                        best_mask = mask

                if best_mask != -1 and best_deg > deg_i:
                    in_family[s_i] = False
                    in_family[best_mask] = True
                    family[i] = best_mask
                    current_score += best_deg - deg_i
                    improved = True
        
        return family, current_score

    family, score = local_search_deadline(family, deadline)
    
    # Convert to lists of ints
    result = []
    for mask in family:
        result.append([i for i in range(n) if (mask >> i) & 1])
        
    return result
