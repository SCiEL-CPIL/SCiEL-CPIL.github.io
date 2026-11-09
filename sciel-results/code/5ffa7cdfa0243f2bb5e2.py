import time
import random

def construct():
    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    
    def get_sums(S):
        sums = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        return sums

    def is_valid_addition(x, S, sums):
        for y in S:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def randomized_greedy_expand(S):
        S_list = list(S)
        S_set = set(S_list)
        sums = get_sums(S_list)
        
        # We try to add elements. To make it randomized, we don't just iterate 1..284.
        # We iterate 1..284, but if multiple are valid, we might pick randomly?
        # Actually, picking the smallest is a good heuristic for compactness.
        # Randomness should come from the perturbation.
        # However, deterministic greedy can get stuck.
        # Let's try a mix: 50% smallest, 50% random from top 10 valid.
        
        for x in range(1, 285):
            if x in S_set:
                continue
            if is_valid_addition(x, S_list, sums):
                # Add it
                S_list.append(x)
                S_set.add(x)
                # Update sums
                for y in S_list[:-1]: # All previous elements
                    sums.add(x + y)
                sums.add(2 * x)
                
        return S_list

    # Initial Greedy
    best_set = randomized_greedy_expand(prescribed)
    
    start_time = time.time()
    
    while time.time() - start_time < 100:
        # Perturb best_set
        S_cur = list(best_set)
        non_prescribed = [x for x in S_cur if x not in prescribed_set]
        
        if non_prescribed:
            # Remove 1 to 3 random non-prescribed elements
            k = random.randint(1, 3)
            if len(non_prescribed) < k:
                k = len(non_prescribed)
            removals = random.sample(non_prescribed, k)
            S_pert = [x for x in S_cur if x not in removals]
        else:
            S_pert = S_cur
            
        # Expand
        S_new = randomized_greedy_expand(S_pert)
        
        if len(S_new) > len(best_set):
            best_set = S_new
            if len(best_set) == 20:
                break
                
    return best_set
