import time
import random
from bisect import bisect_right

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    # Safe base set: (2n/3, n]
    # 2*256/3 = 170.66 -> 171
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
                lst = rem_groups[target_rem]
                # Binary search for first element > b
                idx = bisect_right(lst, b)
                if idx < len(lst):
                    return False
        return True

    def try_add(a, S):
        if a in S:
            return False
        if can_add(a, S):
            S.add(a)
            return True
        return False

    # Phase 1: Greedy Addition
    # Add elements from top to bottom.
    # Larger elements are 'easier' to add (fewer multiples in range).
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline - 15.0:
            break
        try_add(a, S)
    
    # Phase 2: Randomized Local Search (Additions)
    # Try to add remaining elements in random order to catch order-dependent opportunities.
    for _ in range(10):
        if time.time() > deadline - 15.0:
            break
        remaining = [a for a in range(1, base_start) if a not in S]
        if not remaining:
            break
        random.shuffle(remaining)
        for a in remaining:
            if time.time() > deadline - 15.0:
                break
            try_add(a, S)

    # Phase 3: Swap-based Local Search
    # Attempt to remove one element and add two others to increase cardinality.
    # This helps escape local maxima where no single element can be added.
    while time.time() < deadline - 2.0:
        S_list = list(S)
        if len(S_list) < 3:
            break
        
        # Pick a random element to potentially remove
        i = random.randint(0, len(S_list) - 1)
        elem_to_remove = S_list[i]
        
        S_temp = S - {elem_to_remove}
        added_count = 0
        
        # Candidates to add: all elements not in S_temp
        remaining = [a for a in range(1, n + 1) if a not in S_temp]
        random.shuffle(remaining)
        
        # Try to add elements until we have added 2 or time runs out
        for a in remaining:
            if time.time() > deadline - 2.0:
                break
            if a in S_temp:
                continue
            if can_add(a, S_temp):
                S_temp.add(a)
                added_count += 1
                if added_count >= 2:
                    break
            
        # Accept the swap if the set size increased
        if len(S_temp) > len(S):
            S = S_temp

    return list(S)
