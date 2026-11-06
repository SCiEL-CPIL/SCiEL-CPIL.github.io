import time
import random
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    # Safe base set: (2n/3, n]
    base_start = 171
    S = set(range(base_start, n + 1))
    
    # Precompute target remainders: target_rem[a][b] = (-b) % a for b > a
    target_rem = [[0] * (n + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            target_rem[a][b] = (-b) % a
    
    def can_add(a, S):
        """Check if adding a to S violates the property."""
        # Build sorted list of S elements > a, grouped by remainder mod a
        s_elements = [x for x in S if x > a]
        if not s_elements:
            return True
        
        # Group by remainder
        rem_groups = {}
        for x in s_elements:
            r = x % a
            if r not in rem_groups:
                rem_groups[r] = []
            rem_groups[r].append(x)
        
        # Sort each group
        for r in rem_groups:
            rem_groups[r].sort()
        
        # For each b in S with b > a, check conflict
        for b in s_elements:
            tr = target_rem[a][b]
            if tr in rem_groups:
                group = rem_groups[tr]
                idx = bisect_right(group, b)
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
            
            # Try each possible removal
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
            
            # Also try 2-for-3 swaps
            if not improved and time.time() < deadline - 3.0:
                S_list2 = sorted(S)
                not_in_S2 = [a for a in range(1, n + 1) if a not in S]
                random.shuffle(not_in_S2)
                
                # Try removing two elements
                for i in range(min(len(S_list2), 50)):
                    if time.time() > deadline - 3.0:
                        break
                    for j in range(i + 1, min(len(S_list2), 50)):
                        if time.time() > deadline - 3.0:
                            break
                        S_test = S - {S_list2[i], S_list2[j]}
                        added_count = 0
                        added_elems = []
                        
                        for add_elem in not_in_S2:
                            if time.time() > deadline - 3.0:
                                break
                            if can_add(add_elem, S_test):
                                S_test.add(add_elem)
                                added_elems.append(add_elem)
                                added_count += 1
                                if added_count >= 3:
                                    break
                        
                        if added_count > 2:
                            S = S_test
                            improved = True
                            break
                    if improved:
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
    
    # One more round of 1-for-2 swaps with fresh randomization
    if time.time() < deadline - 1.5:
        S_list_final = sorted(S)
        not_in_S_final = [a for a in range(1, n + 1) if a not in S]
        random.shuffle(not_in_S_final)
        
        for rem_elem in S_list_final:
            if time.time() > deadline - 1.0:
                break
            S_test = S - {rem_elem}
            added_count = 0
            added_elems = []
            
            for add_elem in not_in_S_final:
                if time.time() > deadline - 1.0:
                    break
                if can_add(add_elem, S_test):
                    S_test.add(add_elem)
                    added_elems.append(add_elem)
                    added_count += 1
                    if added_count >= 2:
                        break
            
            if added_count > 1:
                S = S_test
    
    return list(S)
