import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 256
    
    # Helper to check if x can be added to S (given as set and sorted list)
    def can_add(x, S_set, S_list):
        # Case 1: x is 'a' (smallest in the triple)
        # Need b, c in S such that x < b <= c and x | (b+c)
        for b in S_list:
            if b <= x:
                continue
            # We need c in S such that c >= b and (b+c) % x == 0
            # c % x == (-b) % x
            target_rem = (-b) % x
            # Iterate c in S_list. Since S_list is sorted, we can break if c < b? 
            # No, we need c >= b. We can start from the position of b.
            # But simple iteration is fine for small N.
            for c in S_list:
                if c < b:
                    continue
                if c % x == target_rem:
                    return False
        # Case 2: x is 'b' (middle)
        # Need a, c in S such that a < x <= c and a | (x+c)
        for a in S_list:
            if a >= x:
                break # S_list is sorted
            # We need c in S such that c >= x and (x+c) % a == 0
            # c % a == (-x) % a
            target_rem = (-x) % a
            for c in S_list:
                if c < x:
                    continue
                if c % a == target_rem:
                    return False
        # Case 3: x is 'c' (largest)
        # Need a, b in S such that a < b < x and a | (b+x)
        # Note: b must be strictly less than x because x is not in S yet
        for a in S_list:
            if a >= x:
                break
            # We need b in S such that a < b < x and (b+x) % a == 0
            # b % a == (-x) % a
            target_rem = (-x) % a
            for b in S_list:
                if b <= a:
                    continue
                if b >= x:
                    break
                if b % a == target_rem:
                    return False
        return True

    def greedy_add(S_set, S_list, order):
        S_set = S_set.copy()
        S_list = S_list.copy()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S_set and can_add(x, S_set, S_list):
                S_set.add(x)
                # Insert into sorted list
                # Simple insertion for small lists
                S_list.append(x)
                S_list.sort()
        return S_set, S_list

    best_set = set()
    best_list = []
    best_size = 0
    
    def update_best(S_set, S_list):
        nonlocal best_set, best_list, best_size
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = S_set.copy()
            best_list = S_list.copy()

    # 1. Greedy Initializations
    # Descending
    S1, L1 = greedy_add(set(), [], range(n, 0, -1))
    update_best(S1, L1)
    
    # Ascending
    S2, L2 = greedy_add(set(), [], range(1, n + 1))
    update_best(S2, L2)
    
    # Random
    for _ in range(5):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_r, L_r = greedy_add(set(), [], order)
        update_best(S_r, L_r)

    # 2. Local Search
    def local_search(S_init, L_init, max_time=5.0):
        S = S_init.copy()
        L = L_init.copy()
        local_deadline = time.time() + max_time
        improved = True
        
        while improved and time.time() < local_deadline:
            improved = False
            
            # Phase 1: Add any element
            candidates = [x for x in range(1, n + 1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > local_deadline:
                    break
                if can_add(x, S, L):
                    S.add(x)
                    L.append(x)
                    L.sort()
                    improved = True
                    break
            
            # Phase 2: 1-for-k swap (Remove 1, Add many)
            if not improved and time.time() < local_deadline:
                S_list_temp = list(S)
                random.shuffle(S_list_temp)
                for e in S_list_temp:
                    if time.time() > local_deadline:
                        break
                    S_temp = S.copy()
                    S_temp.remove(e)
                    L_temp = [x for x in L if x != e]
                    
                    # Try to add as many as possible
                    candidates = [x for x in range(1, n + 1) if x not in S_temp]
                    random.shuffle(candidates)
                    added = 0
                    for x in candidates:
                        if time.time() > local_deadline:
                            break
                        if can_add(x, S_temp, L_temp):
                            S_temp.add(x)
                            L_temp.append(x)
                            L_temp.sort()
                            added += 1
                    
                    if len(S_temp) > len(S):
                        S = S_temp
                        L = L_temp
                        improved = True
                        break
            
            # Phase 3: 2-for-k swap (Remove 2, Add many)
            if not improved and time.time() < local_deadline:
                S_list_temp = list(S)
                random.shuffle(S_list_temp)
                # Limit pairs to avoid O(N^2) explosion in local search step
                limit = min(len(S_list_temp), 15)
                for i in range(limit):
                    if time.time() > local_deadline:
                        break
                    for j in range(i + 1, limit):
                        if time.time() > local_deadline:
                            break
                        e1, e2 = S_list_temp[i], S_list_temp[j]
                        S_temp = S.copy()
                        S_temp.remove(e1)
                        S_temp.remove(e2)
                        L_temp = [x for x in L if x != e1 and x != e2]
                        
                        candidates = [x for x in range(1, n + 1) if x not in S_temp]
                        random.shuffle(candidates)
                        
                        for x in candidates:
                            if time.time() > local_deadline:
                                break
                            if can_add(x, S_temp, L_temp):
                                S_temp.add(x)
                                L_temp.append(x)
                                L_temp.sort()
                        
                        if len(S_temp) > len(S):
                            S = S_temp
                            L = L_temp
                            improved = True
                            break
                    if improved:
                        break
        
        return S, L

    # Run local search from best and random seeds
    iterations = 0
    while time.time() < deadline - 3:
        iterations += 1
        
        # Start from current best
        S_curr, L_curr = local_search(best_set, best_list, max_time=3.0)
        update_best(S_curr, L_curr)
        
        # Periodically restart from random seeds
        if iterations % 3 == 0:
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_rand, L_rand = greedy_add(set(), [], order)
            S_curr, L_curr = local_search(S_rand, L_rand, max_time=3.0)
            update_best(S_curr, L_curr)

    return sorted(best_set)
