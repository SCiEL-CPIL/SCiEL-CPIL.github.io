def construct():
    import time
    import random
    import math
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    
    def is_valid(S_set, S_list):
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        return False
        return True
    
    def count_violations(S_set, S_list):
        m = len(S_list)
        count = 0
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        count += 1
        return count
    
    def violation_count_for_element(S_set, S_list, elem):
        """Count how many violations involve this specific element"""
        m = len(S_list)
        count = 0
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        if elem in (a, b, c):
                            count += 1
        return count
    
    # Start with (128, 256]
    S_list = list(range(129, 257))
    S_set = set(S_list)
    
    # Phase 1: Remove elements to fix violations (same as before)
    while True:
        if time.time() > deadline:
            break
        if is_valid(S_set, S_list):
            break
        from collections import defaultdict
        viol_count = defaultdict(int)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                k_min = max(3, (2*b + a - 1) // a)
                k_max = min(512 // a, (256 + b) // a)
                for k in range(k_min, k_max + 1):
                    c = k * a - b
                    if c >= b and c in S_set:
                        viol_count[a] += 1
                        viol_count[b] += 1
                        viol_count[c] += 1
        worst = max(viol_count, key=viol_count.get)
        S_list.remove(worst)
        S_set.discard(worst)
    
    best_size = len(S_list)
    best_S_list = list(S_list)
    best_S_set = set(S_list)
    
    # Phase 2: Simulated annealing with swap operations
    # Objective: maximize |S| - penalty * violations
    # We'll use swaps: remove one element, add another
    
    all_elements = list(range(1, 257))
    current_S_list = list(S_list)
    current_S_set = set(S_list)
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 5 and T > T_end:
        if no_improve_count > max_no_improve:
            break
            
        # Choose a random operation: add, remove, or swap
        op = random.random()
        
        if op < 0.3 and len(current_S_list) < 256:
            # Try to add a random element
            candidates = [x for x in all_elements if x not in current_S_set]
            if not candidates:
                T *= cooling_rate
                continue
            x = random.choice(candidates)
            new_S_list = current_S_list + [x]
            new_S_list.sort()
            new_S_set = current_S_set | {x}
            
            old_viol = count_violations(current_S_set, current_S_list)
            new_viol = count_violations(new_S_set, new_S_list)
            
            old_score = len(current_S_list) - old_viol
            new_score = len(new_S_list) - new_viol
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_S_list = new_S_list
                current_S_set = new_S_set
                if len(current_S_list) > best_size:
                    best_size = len(current_S_list)
                    best_S_list = list(current_S_list)
                    best_S_set = set(current_S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif op < 0.5 and len(current_S_list) > 1:
            # Try to remove a random element
            x = random.choice(current_S_list)
            new_S_list = [y for y in current_S_list if y != x]
            new_S_set = current_S_set - {x}
            
            old_viol = count_violations(current_S_set, current_S_list)
            new_viol = count_violations(new_S_set, new_S_list)
            
            old_score = len(current_S_list) - old_viol
            new_score = len(new_S_list) - new_viol
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_S_list = new_S_list
                current_S_set = new_S_set
                if len(current_S_list) > best_size:
                    best_size = len(current_S_list)
                    best_S_list = list(current_S_list)
                    best_S_set = set(current_S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        else:
            # Try a swap: remove one, add another
            if len(current_S_list) == 0:
                T *= cooling_rate
                continue
            candidates = [x for x in all_elements if x not in current_S_set]
            if not candidates:
                T *= cooling_rate
                continue
            x_remove = random.choice(current_S_list)
            x_add = random.choice(candidates)
            
            new_S_list = [y for y in current_S_list if y != x_remove] + [x_add]
            new_S_list.sort()
            new_S_set = (current_S_set - {x_remove}) | {x_add}
            
            old_viol = count_violations(current_S_set, current_S_list)
            new_viol = count_violations(new_S_set, new_S_list)
            
            old_score = len(current_S_list) - old_viol
            new_score = len(new_S_list) - new_viol
            delta = new_score - old_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_S_list = new_S_list
                current_S_set = new_S_set
                if len(current_S_list) > best_size:
                    best_size = len(current_S_list)
                    best_S_list = list(current_S_list)
                    best_S_set = set(current_S_list)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        T *= cooling_rate
    
    # Phase 3: Try to add back elements greedily
    removed = set(range(1, 257)) - best_S_set
    for a in sorted(removed, reverse=True):
        if time.time() > deadline - 2:
            break
        S_list_test = best_S_list + [a]
        S_list_test.sort()
        S_set_test = best_S_set | {a}
        if is_valid(S_set_test, S_list_test):
            best_S_list = S_list_test
            best_S_set = S_set_test
    
    return best_S_list
