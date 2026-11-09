import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 88.0
    
    def can_add(S_set, x):
        """Check if adding x to S_set keeps it valid. O(k^2) worst case."""
        if x in S_set:
            return True
        
        S_list = sorted(S_set)
        k = len(S_list)
        
        # Precompute for efficiency
        # Case 1: a=x, need b,c in S with x < b <= c and x ∤ (b+c)
        # For each pair (b,c) with b,c > x, check (b+c) % x != 0
        for i in range(k):
            b = S_list[i]
            if b <= x:
                continue
            for j in range(i, k):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: a in S, a < x, c in S, c >= x, check a ∤ (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c >= x and (x + c) % a == 0:
                    return False
        
        # Case 3: a,b in S, a < b <= x, check a ∤ (b+x)
        for i in range(k):
            a = S_list[i]
            for j in range(i+1, k):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        return True
    
    def full_validate(S):
        """Full O(k^3) validation."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            for j in range(i+1, k):
                for l in range(j, k):
                    a, b, c = S_sorted[i], S_sorted[j], S_sorted[l]
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    best_set = set(range(171, 257))  # Known valid, 86 elements
    best_size = len(best_set)
    
    # Try different starting tails
    for L in range(155, 172):
        if time.time() > deadline - 5:
            break
        S = set(range(L, 257))
        if not full_validate(S):
            continue
        if len(S) > best_size:
            best_set = set(S)
            best_size = len(S)
    
    # Greedy addition: try adding elements from 1 to L-1 in various orders
    for L in range(155, 172):
        if time.time() > deadline - 12:
            break
        base = set(range(L, 257))
        if not full_validate(base):
            continue
        
        # Ascending order
        S = set(base)
        for x in range(1, L):
            if time.time() > deadline - 5:
                break
            if can_add(S, x):
                S.add(x)
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
        
        # Descending order
        S = set(base)
        for x in range(L-1, 0, -1):
            if time.time() > deadline - 5:
                break
            if can_add(S, x):
                S.add(x)
        if len(S) > best_size and full_validate(S):
            best_set = set(S)
            best_size = len(S)
        
        # Random shuffled orders (a few)
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            candidates = list(range(1, L))
            random.shuffle(candidates)
            S = set(base)
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(S, x):
                    S.add(x)
            if len(S) > best_size and full_validate(S):
                best_set = set(S)
                best_size = len(S)
    
    # Try local search: swap elements
    if time.time() < deadline - 12:
        S = set(best_set)
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            S_list = sorted(S)
            for i, elem in enumerate(S_list):
                if time.time() > deadline - 8:
                    break
                for x in range(1, n+1):
                    if x in S:
                        continue
                    S.remove(elem)
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
                    S.add(elem)
                if time.time() > deadline - 8:
                    break
        
        # Second pass: try adding elements without removing (pure addition)
        if time.time() < deadline - 5:
            S_list = sorted(S)
            for x in range(1, n+1):
                if time.time() > deadline - 5:
                    break
                if x in S:
                    continue
                if can_add(S, x):
                    S.add(x)
            if len(S) > best_size and full_validate(S):
                best_set = set(S)
                best_size = len(S)
    
    # Final validation
    if not full_validate(best_set):
        # Fallback to known valid set
        best_set = set(range(171, 257))
    
    return list(best_set)
