import time
import random
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 82.0
    
    n = 256
    # Safe base set: (2n/3, n]
    base_start = 171
    S = set(range(base_start, n + 1))
    
    def can_add(a, S):
        # Get elements in S greater than a, sorted
        elems = sorted(x for x in S if x > a)
        if len(elems) < 2:
            return True
        
        # Group elements by their remainder mod a
        rem_groups = {}
        for e in elems:
            r = e % a
            if r not in rem_groups:
                rem_groups[r] = []
            rem_groups[r].append(e)
        
        # For each b in elems, check if there exists c > b with c % a == (-b) % a
        for b in elems:
            target_rem = (-b) % a
            if target_rem in rem_groups:
                # Binary search for first element > b in rem_groups[target_rem]
                group = rem_groups[target_rem]
                idx = bisect_left(group, b + 1)
                if idx < len(group):
                    return False
        return True

    # Greedy Phase: Add from top to bottom
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Local Search: Try to add remaining elements in random order
    remaining = [a for a in range(1, base_start) if a not in S]
    random.shuffle(remaining)
    
    for a in remaining:
        if time.time() > deadline:
            break
        if can_add(a, S):
            S.add(a)
    
    # Multiple passes to catch order-dependent additions
    for _ in range(3):
        if time.time() > deadline:
            break
        remaining2 = [a for a in range(1, n + 1) if a not in S]
        if not remaining2:
            break
        random.shuffle(remaining2)
        for a in remaining2:
            if time.time() > deadline:
                break
            if can_add(a, S):
                S.add(a)
    
    # Swap-based local search: try removing one element to allow adding two others
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            S_list = sorted(S)
            not_in_S = [a for a in range(1, n + 1) if a not in S]
            random.shuffle(not_in_S)
            
            for rem_elem in S_list:
                if time.time() > deadline - 2.0:
                    break
                # Try removing rem_elem
                S_test = S - {rem_elem}
                added_count = 0
                added_elems = []
                
                for add_elem in not_in_S:
                    if time.time() > deadline - 2.0:
                        break
                    if can_add(add_elem, S_test):
                        S_test.add(add_elem)
                        added_elems.append(add_elem)
                        added_count += 1
                        if added_count >= 2:
                            break
                
                if added_count > 1:
                    S = S_test
                    improved = True
                    break
    
    # Final pass
    if time.time() < deadline - 1.0:
        remaining3 = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(remaining3)
        for a in remaining3:
            if time.time() > deadline - 0.5:
                break
            if can_add(a, S):
                S.add(a)
    
    return list(S)
