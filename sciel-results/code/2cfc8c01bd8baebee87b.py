"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    deadline = time.time() + 55  # 5s safety margin
    
    def get_sum(indices):
        s = Fraction(0)
        for i in indices:
            s += reciprocals[i]
        return s

    # 1. Greedy Initialization
    best_indices = set()
    best_sum = Fraction(0)
    for i in range(n):
        if best_sum + reciprocals[i] < 1:
            best_indices.add(i)
            best_sum += reciprocals[i]
    
    # 2. Local Search
    def local_search(start_indices):
        current_indices = set(start_indices)
        current_sum = get_sum(current_indices)
        
        while True:
            improved = False
            
            # Phase 1: Add any element that fits
            best_add_idx = -1
            best_add_sum = current_sum
            for j in range(n):
                if j not in current_indices:
                    new_sum = current_sum + reciprocals[j]
                    if new_sum < 1 and new_sum > best_add_sum:
                        best_add_sum = new_sum
                        best_add_idx = j
            
            if best_add_idx != -1:
                current_indices.add(best_add_idx)
                current_sum = best_add_sum
                continue # Restart loop to check for more additions
                
            # Phase 2: Swap one out, one in
            best_swap_in = -1
            best_swap_out = -1
            best_swap_sum = current_sum
            
            for i in current_indices:
                for j in range(n):
                    if j not in current_indices and reciprocals[j] > reciprocals[i]:
                        new_sum = current_sum - reciprocals[i] + reciprocals[j]
                        if new_sum < 1 and new_sum > best_swap_sum:
                            best_swap_sum = new_sum
                            best_swap_in = j
                            best_swap_out = i
                            
            if best_swap_in != -1:
                current_indices.remove(best_swap_out)
                current_indices.add(best_swap_in)
                current_sum = best_swap_sum
                continue # Restart loop
                
            # Phase 3: Double Swap
            best_dswap_remove = None
            best_dswap_add = None
            best_dswap_sum = current_sum
            
            removals = list(current_indices)
            m = len(removals)
            
            if m >= 2:
                for a_idx in range(m):
                    for b_idx in range(a_idx + 1, m):
                        i1 = removals[a_idx]
                        i2 = removals[b_idx]
                        base_sum = current_sum - reciprocals[i1] - reciprocals[i2]
                        
                        for j1 in range(n):
                            if j1 in current_indices: continue
                            for j2 in range(j1 + 1, n):
                                if j2 in current_indices: continue
                                
                                add_val = reciprocals[j1] + reciprocals[j2]
                                new_sum = base_sum + add_val
                                if new_sum < 1 and new_sum > best_dswap_sum:
                                    best_dswap_sum = new_sum
                                    best_dswap_remove = (i1, i2)
                                    best_dswap_add = (j1, j2)
                                    
            if best_dswap_remove is not None:
                current_indices.discard(best_dswap_remove[0])
                current_indices.discard(best_dswap_remove[1])
                current_indices.add(best_dswap_add[0])
                current_indices.add(best_dswap_add[1])
                current_sum = best_dswap_sum
                continue # Restart loop

            break
            
        return current_indices, current_sum

    # Run local search from greedy
    local_indices, local_sum = local_search(best_indices)
    if local_sum > best_sum:
        best_indices = local_indices
        best_sum = local_sum
        
    # Random Restarts
    random.seed(42)
    
    while time.time() < deadline:
        k = random.randint(3, 10)
        random_indices = random.sample(range(n), k)
        
        r_indices = set(random_indices)
        r_sum = get_sum(r_indices)
        
        sorted_r = sorted(r_indices, key=lambda x: reciprocals[x], reverse=True)
        for idx in sorted_r:
            if r_sum < 1:
                break
            r_sum -= reciprocals[idx]
            r_indices.remove(idx)
            
        if not r_indices:
            continue
            
        l_indices, l_sum = local_search(r_indices)
        
        if l_sum > best_sum:
            best_indices = l_indices
            best_sum = l_sum
            
    return list(best_indices)
# EVOLVE-BLOCK-END
