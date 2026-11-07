import time
import random
import sys

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin
    
    N = 12
    M = 40
    NUM_SUBSETS = 1 << N
    
    # Precompute comparability sets
    comp_sets = [set() for _ in range(NUM_SUBSETS)]
    masks = list(range(NUM_SUBSETS))
    
    # Precomputation
    for i in range(NUM_SUBSETS):
        mi = masks[i]
        for j in range(i + 1, NUM_SUBSETS):
            mj = masks[j]
            if (mi & mj) == mi or (mi & mj) == mj:
                comp_sets[mi].add(mj)
                comp_sets[mj].add(mi)
                
    def greedy_construct():
        """
        Greedily construct a set of M masks, starting with 0 and full set.
        """
        current_set = {0, NUM_SUBSETS - 1}
        
        for _ in range(M - 2):
            best_mask = -1
            best_score = -1
            
            for x in masks:
                if x in current_set:
                    continue
                score = len(current_set & comp_sets[x])
                if score > best_score:
                    best_score = score
                    best_mask = x
            
            if best_mask != -1:
                current_set.add(best_mask)
            else:
                break
                
        return list(current_set)

    def local_search(initial_list):
        """
        Improve the set using local search.
        """
        current_set = set(initial_list)
        
        # Calculate initial score
        def get_score(s):
            total_deg = 0
            for x in s:
                total_deg += len(s & comp_sets[x])
            return total_deg // 2
        
        current_score = get_score(current_set)
        
        improved = True
        # We run multiple passes until no improvement or time limit
        while improved:
            improved = False
            items = list(current_set)
            random.shuffle(items)
            
            for old in items:
                if time.time() - start_time > time_limit - 5.0:
                    return list(current_set), current_score
                
                current_set.remove(old)
                
                best_new = -1
                best_new_score = -1
                deg_old = len(current_set & comp_sets[old])
                
                for x in masks:
                    if x in current_set:
                        continue
                    deg_new = len(current_set & comp_sets[x])
                    if deg_new > best_new_score:
                        best_new_score = deg_new
                        best_new = x
                        
                if best_new != -1 and best_new_score > deg_old:
                    current_set.add(best_new)
                    current_score += (best_new_score - deg_old)
                    improved = True
                else:
                    current_set.add(old)
                    
        return list(current_set), current_score

    best_solution = None
    best_score = -1
    
    # Run multiple restarts
    while time.time() - start_time < time_limit - 10.0:
        init_list = greedy_construct()
        final_list, score = local_search(init_list)
        
        if score > best_score:
            best_score = score
            best_solution = final_list
            
        if best_score > 700:
            break

    # Convert masks to lists of ints
    result = []
    for msk in best_solution:
        s = []
        for i in range(N):
            if msk & (1 << i):
                s.append(i)
        result.append(s)
        
    return result
