import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    candidates = list(range(2, 257))
    N = 256
    
    # Precompute gcds and divisibility helpers
    # divisors[k] = list of numbers in 2..256 divisible by k
    # Actually, we need to check against the current set S.
    
    # Helper to check if x can be added to S
    # S is a list of ints
    def can_add(x, S):
        if len(S) < 2:
            return True
        
        # Check 1: x does not divide b*c for any distinct b,c in S
        # Check 2: b does not divide x*c for any distinct b,c in S
        # Check 3: c does not divide x*b for any distinct b,c in S (Symmetric to 2)
        
        # Optimization:
        # For Check 1: x | bc.
        # For Check 2/3: b | xc or c | xb.
        # b | xc <=> (b/gcd(b,x)) | c.
        
        # We can iterate over b in S
        for b in S:
            # Check if b | x*c for some c in S, c != b
            g = gcd(b, x)
            b_reduced = b // g
            # We need c in S, c != b, such that c % b_reduced == 0
            # If b_reduced == 1, then any c != b works? No, c must be in S.
            # If b_reduced > 1, we check multiples.
            
            # To make this fast, we can precompute for the current S:
            # multiples_count[k] = number of elements in S divisible by k
            # But S changes.
            
            # For now, let's just iterate c.
            # If S is small, this is fine.
            # But S can be ~100.
            # 100 * 100 = 10,000 ops per can_add.
            
            found_c = False
            for c in S:
                if c == b:
                    continue
                if c % b_reduced == 0:
                    found_c = True
                    break
            if found_c:
                return False
        
        # Check 1: x | b*c
        for i in range(len(S)):
            b = S[i]
            for j in range(i + 1, len(S)):
                c = S[j]
                if (b * c) % x == 0:
                    return False
            
        return True

    best_set = []
    
    def run_greedy(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        return S

    # Greedy strategies
    order1 = sorted(candidates, reverse=True)
    set1 = run_greedy(order1)
    if len(set1) > len(best_set):
        best_set = set1

    order2 = sorted(candidates, key=lambda x: (-x.bit_count(), -x))
    set2 = run_greedy(order2)
    if len(set2) > len(best_set):
        best_set = set2

    # Local Search Refinement
    # Try to improve best_set
    S = best_set[:]
    S_set = set(S)
    
    # Precompute multiples for S to speed up can_add?
    # Let's just use the O(N^2) can_add for now, it's probably fast enough for a few iterations.
    
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        # Phase 1: Add
        # Sort candidates not in S by descending value
        candidates_to_add = [x for x in candidates if x not in S_set]
        candidates_to_add.sort(reverse=True)
        
        for x in candidates_to_add:
            if time.time() > deadline - 5:
                break
            if can_add(x, S):
                S.append(x)
                S_set.add(x)
                improved = True
                # Don't break, keep adding if possible
        
        if improved:
            continue
            
        # Phase 2: Swaps
        # Try to swap y in S with x not in S
        # To avoid O(N^3), we limit the number of swaps tried or use randomization
        S_list = S[:]
        S_list.sort(key=lambda y: -y) # Try removing large numbers first? Or small?
        # Removing small numbers might be better as they divide more things?
        # Actually, small numbers are more "constraining". Removing them might free up space.
        
        candidates_to_swap_in = [x for x in candidates if x not in S_set]
        candidates_to_swap_in.sort(reverse=True)
        
        # Limit swaps to prevent timeout
        swaps_tried = 0
        max_swaps_per_iter = 200
        
        for y in S_list:
            if swaps_tried >= max_swaps_per_iter:
                break
            if time.time() > deadline - 5:
                break
            
            S_temp_set = S_set - {y}
            S_temp_list = [z for z in S if z != y]
            
            for x in candidates_to_swap_in:
                if swaps_tried >= max_swaps_per_iter:
                    break
                swaps_tried += 1
                
                # Check if x can be added to S_temp
                # Reusing can_add logic but with S_temp_list
                # This is expensive.
                # Let's just do a simple check.
                
                # If x is in S_temp_set, skip (shouldn't happen)
                if x in S_temp_set:
                    continue
                    
                # Check validity of S_temp_list + [x]
                # We can use can_add(x, S_temp_list)
                if can_add(x, S_temp_list):
                    # Perform swap
                    S.remove(y)
                    S.append(x)
                    S_set.remove(y)
                    S_set.add(x)
                    improved = True
                    break # Restart inner loop? Or just continue?
                    # Breaking inner loop and restarting outer is safer for convergence
                    break
            
            if improved:
                break # Restart outer loop
        
    return [int(x) for x in S]
