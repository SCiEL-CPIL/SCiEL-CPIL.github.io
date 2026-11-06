# EVOLVE-BLOCK-START
import time
import random
import math
from functools import reduce
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin: 85s limit to avoid 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    if hasattr(math, 'lcm'):
        L = math.lcm(*denominators)
    else:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        L = reduce(lambda a, b: a * b // gcd(a, b), denominators)
        
    inv_int = [L // d for d in denominators]
    
    # 1. Greedy baseline: pick in order while sum < L
    selected = set()
    total_int = 0
    for i in range(n):
        if total_int + inv_int[i] < L:
            selected.add(i)
            total_int += inv_int[i]
    
    best_selected = set(selected)
    best_total_int = total_int
    best_deficit_int = L - total_int
    
    # Helper to check if a set is valid and update best
    def try_update(sel, tot_int):
        nonlocal best_selected, best_total_int, best_deficit_int
        if tot_int < L and tot_int > best_total_int:
            best_total_int = tot_int
            best_selected = set(sel)
            best_deficit_int = L - tot_int
            return True
        return False
    
    # Fast local search using integer arithmetic
    def fast_local_search(start_sel, start_tot_int, time_limit):
        nonlocal best_selected, best_total_int
        selected = set(start_sel)
        total_int = start_tot_int
        
        if time.time() >= time_limit:
            return selected, total_int
            
        while time.time() < time_limit:
            improved = False
            
            # 1. Try inserting any unselected element
            for i in range(n):
                if i not in selected:
                    new_total = total_int + inv_int[i]
                    if new_total < L:
                        if new_total > best_total_int:
                            if try_update(selected | {i}, new_total):
                                selected = best_selected
                                total_int = best_total_int
                                improved = True
            
            # 2. Try swapping any selected with any unselected
            sel_list = list(selected)
            for i in sel_list:
                if time.time() >= time_limit:
                    return selected, total_int
                val_i = inv_int[i]
                for j in range(n):
                    if j not in selected:
                        val_j = inv_int[j]
                        if val_j <= val_i:
                            continue
                        new_total = total_int - val_i + val_j
                        if new_total < L:
                            if new_total > best_total_int:
                                if try_update((selected - {i}) | {j}, new_total):
                                    selected = best_selected
                                    total_int = best_total_int
                                    improved = True
            
            if not improved:
                break
        
        return selected, total_int

    # Run fast local search from greedy
    fast_local_search(best_selected, best_total_int, deadline - 10)
    
    # Perturbation-based restarts
    while time.time() < deadline - 5:
        start_sel = set(best_selected)
        start_tot_int = best_total_int
        
        # Perturb
        K = random.randint(2, 5)
        to_remove = random.sample(list(start_sel), min(K, len(start_sel)))
        for i in to_remove:
            start_sel.remove(i)
            start_tot_int -= inv_int[i]
            
        available = [i for i in range(n) if i not in start_sel]
        if available:
            to_add = random.sample(available, min(K, len(available)))
            for i in to_add:
                if start_tot_int + inv_int[i] < L:
                    start_sel.add(i)
                    start_tot_int += inv_int[i]

        if start_tot_int >= L:
            while start_tot_int >= L and start_sel:
                min_idx = min(start_sel, key=lambda x: inv_int[x])
                start_sel.remove(min_idx)
                start_tot_int -= inv_int[min_idx]
        
        fast_local_search(start_sel, start_tot_int, deadline - 5)
        
    return list(best_selected)
# EVOLVE-BLOCK-END
