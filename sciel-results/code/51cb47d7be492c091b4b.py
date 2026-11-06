import time
from fractions import Fraction
from math import lcm
import random

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    deadline = time.time() + 80  # 10s safety margin
    
    # Compute L and numerators for fast arithmetic
    L = 1
    for d in denominators:
        L = lcm(L, d)
    numerators = [L // d for d in denominators]
    
    def get_sum_int(indices):
        s = 0
        for i in indices:
            s += numerators[i]
        return s
    
    # Greedy initialization using integer arithmetic
    best_indices = set()
    best_sum = 0
    for i in range(n):
        if best_sum + numerators[i] < L:
            best_indices.add(i)
            best_sum += numerators[i]
    
    def local_search(start_indices, deadline):
        current_indices = set(start_indices)
        current_sum = get_sum_int(current_indices)
        
        while time.time() < deadline:
            improved = False
            
            # Phase 1: Add any element that fits
            best_add_idx = -1
            best_add_sum = current_sum
            for j in range(n):
                if j not in current_indices:
                    new_sum = current_sum + numerators[j]
                    if new_sum < L and new_sum > best_add_sum:
                        best_add_sum = new_sum
                        best_add_idx = j
            
            if best_add_idx != -1:
                current_indices.add(best_add_idx)
                current_sum = best_add_sum
                improved = True
                continue
            
            # Phase 2: Swap one out, one in
            best_swap_in = -1
            best_swap_out = -1
            best_swap_sum = current_sum
            
            for i in current_indices:
                for j in range(n):
                    if j not in current_indices and numerators[j] > numerators[i]:
                        new_sum = current_sum - numerators[i] + numerators[j]
                        if new_sum < L and new_sum > best_swap_sum:
                            best_swap_sum = new_sum
                            best_swap_in = j
                            best_swap_out = i
                            
            if best_swap_in != -1:
                current_indices.remove(best_swap_out)
                current_indices.add(best_swap_in)
                current_sum = best_swap_sum
                improved = True
                continue
            
            # Phase 3: Double Swap - optimized
            best_dswap_remove = None
            best_dswap_add = None
            best_dswap_sum = current_sum
            
            removals = list(current_indices)
            m = len(removals)
            
            if m >= 2:
                # Precompute available indices
                available = [j for j in range(n) if j not in current_indices]
                
                for a_idx in range(m):
                    for b_idx in range(a_idx + 1, m):
                        i1 = removals[a_idx]
                        i2 = removals[b_idx]
                        base_sum = current_sum - numerators[i1] - numerators[i2]
                        
                        for j1 in available:
                            for j2 in available:
                                if j1 == j2:
                                    continue
                                add_val = numerators[j1] + numerators[j2]
                                new_sum = base_sum + add_val
                                if new_sum < L and new_sum > best_dswap_sum:
                                    best_dswap_sum = new_sum
                                    best_dswap_remove = (i1, i2)
                                    best_dswap_add = (j1, j2)
                                    
            if best_dswap_remove is not None:
                current_indices.discard(best_dswap_remove[0])
                current_indices.discard(best_dswap_remove[1])
                current_indices.add(best_dswap_add[0])
                current_indices.add(best_dswap_add[1])
                current_sum = best_dswap_sum
                improved = True
                continue
            
            break
            
        return current_indices, current_sum
    
    # Run local search from greedy
    local_indices, local_sum = local_search(best_indices, deadline)
    if local_sum > best_sum:
        best_indices = local_indices
        best_sum = local_sum
        
    # Random Restarts
    random.seed(42)
    
    while time.time() < deadline:
        k = random.randint(3, 15)
        random_indices = random.sample(range(n), k)
        
        r_indices = set(random_indices)
        r_sum = get_sum_int(r_indices)
        
        sorted_r = sorted(r_indices, key=lambda x: numerators[x], reverse=True)
        for idx in sorted_r:
            if r_sum < L:
                break
            r_sum -= numerators[idx]
            r_indices.remove(idx)
            
        if not r_indices:
            continue
            
        l_indices, l_sum = local_search(r_indices, deadline)
        
        if l_sum > best_sum:
            best_indices = l_indices
            best_sum = l_sum
            
    return list(best_indices)
