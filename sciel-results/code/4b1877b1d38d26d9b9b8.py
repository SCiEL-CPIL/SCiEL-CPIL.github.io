import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 60s kill
    
    # 1. Precompute 4-APs
    ap_list = []
    aps_containing = [[] for _ in range(N + 1)]
    
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            d_val = a + 3 * d
            ap_idx = len(ap_list)
            ap_list.append((a, b, c, d_val))
            for v in (a, b, c, d_val):
                aps_containing[v].append(ap_idx)
    
    num_aps = len(ap_list)
    
    # Precompute AP membership as bitsets for fast swap checks
    # For each element, store which APs it belongs to
    # For fast swap validation: for element x, we need to check APs containing x
    # and see if after removing y, count < 3
    
    # 2. Helper functions
    def is_valid_addition(ap_count, x):
        for ap_idx in aps_containing[x]:
            if ap_count[ap_idx] == 3:
                return False
        return True
    
    def greedy_construct(order):
        S = set()
        ap_count = [0] * num_aps
        for x in order:
            if is_valid_addition(ap_count, x):
                S.add(x)
                for ap_idx in aps_containing[x]:
                    ap_count[ap_idx] += 1
        return S, ap_count
    
    def base3_score(x):
        """Score based on base-3 digit sum (lower = better for AP-free)"""
        s = 0
        n = x
        while n > 0:
            s += n % 3
            n //= 3
        return s
    
    def local_search(S, ap_count, deadline):
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        
        all_elements = list(range(1, N + 1))
        S_list = list(S)
        
        while time.time() < deadline:
            progress = False
            
            # Phase 1: Try to add random elements
            for _ in range(200):
                if time.time() >= deadline:
                    break
                x = random.choice(all_elements)
                if x in S:
                    continue
                if is_valid_addition(ap_count, x):
                    S.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count[ap_idx] += 1
                    progress = True
                    if len(S) > best_size:
                        best_S = S.copy()
                        best_ap_count = ap_count.copy()
                        best_size = len(S)
                    break
            
            # Phase 2: Try swaps
            if not progress:
                swapped = False
                for _ in range(100):
                    if time.time() >= deadline:
                        break
                    if not S:
                        break
                    x = random.choice(all_elements)
                    if x in S:
                        continue
                    y = random.choice(S_list)
                    
                    # Check swap validity: S - {y} + {x}
                    valid = True
                    for ap_idx in aps_containing[x]:
                        cnt = ap_count[ap_idx]
                        # Check if y is in this AP
                        a, b, c, d_val = ap_list[ap_idx]
                        y_in_ap = (y == a or y == b or y == c or y == d_val)
                        if cnt - (1 if y_in_ap else 0) >= 3:
                            valid = False
                            break
                    
                    if valid:
                        S.discard(y)
                        if y in S_list:
                            S_list.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        
                        S.add(x)
                        S_list.append(x)
                        for ap_idx in aps_containing[x]:
                            ap_count[ap_idx] += 1
                        
                        swapped = True
                        progress = True
                        break
                
                # Phase 3: Remove and refill
                if not progress:
                    if S:
                        y = random.choice(S_list)
                        S_backup = S.copy()
                        S_list_backup = S_list.copy()
                        ap_count_backup = ap_count.copy()
                        
                        S.discard(y)
                        S_list.remove(y)
                        for ap_idx in aps_containing[y]:
                            ap_count[ap_idx] -= 1
                        
                        # Try to refill with shuffled candidates
                        candidates = [z for z in all_elements if z not in S]
                        random.shuffle(candidates)
                        added_count = 0
                        for z in candidates:
                            if time.time() >= deadline:
                                break
                            if is_valid_addition(ap_count, z):
                                S.add(z)
                                S_list.append(z)
                                for ap_idx in aps_containing[z]:
                                    ap_count[ap_idx] += 1
                                added_count += 1
                        
                        if len(S) > len(S_backup):
                            best_S = S.copy()
                            best_ap_count = ap_count.copy()
                            best_size = len(S)
                            progress = True
                        else:
                            S = S_backup
                            S_list = S_list_backup
                            ap_count = ap_count_backup
        
        return best_S, best_ap_count
    
    def final_polish(S, ap_count, deadline):
        """Deterministic single-swap polish: try all possible swaps"""
        best_S = S.copy()
        best_ap_count = ap_count.copy()
        best_size = len(S)
        
        S_list = list(S)
        all_elements = list(range(1, N + 1))
        
        # Try removing each element and see if we can add more
        for i in range(len(S_list)):
            if time.time() >= deadline:
                break
            y = S_list[i]
            
            S_test = S.copy()
            S_test.discard(y)
            ap_count_test = ap_count.copy()
            for ap_idx in aps_containing[y]:
                ap_count_test[ap_idx] -= 1
            
            # Try to add any element that fits
            added = False
            for x in all_elements:
                if x in S_test:
                    continue
                if is_valid_addition(ap_count_test, x):
                    S_test.add(x)
                    for ap_idx in aps_containing[x]:
                        ap_count_test[ap_idx] += 1
                    added = True
                    break
            
            if added and len(S_test) > best_size:
                best_S = S_test
                best_ap_count = ap_count_test
                best_size = len(S_test)
        
        return best_S, best_ap_count
    
    best_S = set()
    best_size = 0
    
    # Strategy 1: Base-3 ordered greedy (deterministic)
    order_base3 = sorted(range(1, N + 1), key=lambda x: (base3_score(x), x))
    S, ap_count = greedy_construct(order_base3)
    if len(S) > best_size:
        best_S = S.copy()
        best_size = len(S)
    
    # Strategy 2: Multiple random restarts with local search
    num_restarts = 0
    while time.time() < time_limit - 10:
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, ap_count = greedy_construct(order)
        
        if len(S) > best_size:
            best_S = S.copy()
            best_size = len(S)
        
        # Apply local search if time permits
        if time.time() < time_limit - 5:
            S_ls, ap_count_ls = local_search(S, ap_count, time_limit - 5)
            if len(S_ls) > best_size:
                best_S = S_ls
                best_size = len(S_ls)
        
        num_restarts += 1
    
    # Final polish on the best set found
    if time.time() < time_limit - 2:
        # Reconstruct ap_count for best_S
        ap_count_final = [0] * num_aps
        for x in best_S:
            for ap_idx in aps_containing[x]:
                ap_count_final[ap_idx] += 1
        
        S_polished, ap_count_polished = final_polish(best_S, ap_count_final, time_limit)
        if len(S_polished) > best_size:
            best_S = S_polished
            best_size = len(S_polished)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
