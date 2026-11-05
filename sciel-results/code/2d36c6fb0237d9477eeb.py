import time
import random
import bisect

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    def check_valid(S_list):
        """Check if S_list (sorted) satisfies the property."""
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def is_valid_add(S_list, m):
        """Check if adding m to S_list (sorted) maintains the property. Returns True if valid."""
        pos = bisect.bisect_left(S_list, m)
        
        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        for j in range(pos, len(S_list)):
            b = S_list[j]
            for k in range(j, len(S_list)):
                c = S_list[k]
                if (b + c) % m == 0:
                    return False
        
        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a in S with a < m, c in S with c >= m
        for i in range(0, pos):
            a = S_list[i]
            for k in range(pos, len(S_list)):
                c = S_list[k]
                if (m + c) % a == 0:
                    return False
        
        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b in S with b <= m
        for i in range(0, pos):
            a = S_list[i]
            for j in range(i + 1, pos):
                b = S_list[j]
                if (b + m) % a == 0:
                    return False
        
        return True

    def greedy_add(S_list, candidates, order):
        """Try to add candidates in the given order to S_list. Returns the resulting list."""
        S = list(S_list)
        for m in order:
            if time.time() > deadline:
                break
            if is_valid_add(S, m):
                pos = bisect.bisect_left(S, m)
                S.insert(pos, m)
        return S

    def local_search(S_list):
        """Try to add more elements via local search. Returns improved list."""
        improved = True
        S = list(S_list)
        while improved and time.time() < deadline:
            improved = False
            in_S = set(S)
            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
            
            for add_m in not_in_S:
                if time.time() > deadline:
                    break
                if is_valid_add(S, add_m):
                    pos = bisect.bisect_left(S, add_m)
                    S.insert(pos, add_m)
                    improved = True
                    in_S = set(S)
                    not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                    break
        return S

    def swap_search(S_list):
        """Try 1-for-1 swaps to enable further additions. Returns improved list."""
        S = list(S_list)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            in_S = set(S)
            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
            
            for remove_m in list(S):
                if time.time() > deadline:
                    break
                S_test = [x for x in S if x != remove_m]
                for add_m in not_in_S:
                    if time.time() > deadline:
                        break
                    if is_valid_add(S_test, add_m):
                        new_S = S_test + [add_m]
                        new_S.sort()
                        # Check if this swap enables further additions
                        new_S2 = local_search(new_S)
                        if len(new_S2) > len(S):
                            S = new_S2
                            improved = True
                            in_S = set(S)
                            not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                            break
                if improved:
                    break
        return S

    # Strategy: Start with {171,...,256}, try multiple random orders to add elements
    base_set = list(range(171, n + 1))
    best_S = list(base_set)
    best_size = len(best_S)
    
    random.seed(42)
    
    # Multiple random restarts with different addition orders
    for restart in range(60):
        if time.time() > deadline:
            break
        
        S = list(base_set)
        candidates = list(range(1, 171))
        
        if restart % 3 == 0:
            # Ascending order with some randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: random.random())
        elif restart % 3 == 1:
            # Descending order with randomization
            random.shuffle(candidates)
            candidates.sort(key=lambda x: -random.random())
        else:
            # Pure random order
            random.shuffle(candidates)
        
        S = greedy_add(S, candidates, candidates)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Local search: try to add more elements
    best_S = local_search(best_S)
    best_size = len(best_S)
    
    # Swap search: try to improve via swaps
    best_S = swap_search(best_S)
    best_size = len(best_S)
    
    # Final verification
    S_final = sorted(best_S)
    if not check_valid(S_final):
        # Fallback to known valid set
        return list(range(171, 257))
    
    return S_final
