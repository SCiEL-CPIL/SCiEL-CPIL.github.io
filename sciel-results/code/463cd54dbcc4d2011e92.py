import time
from itertools import combinations

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    def check_valid(S_set):
        """Check if S_set satisfies the property."""
        S_list = sorted(S_set)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for c_val in S_list[j:]:
                    c = c_val
                    if (b + c) % a == 0:
                        return False
        return True
    
    def is_valid_add(S_set, m):
        """Check if adding m to S_set maintains the property. S_set is a set."""
        S_list = sorted(S_set)
        
        # Case 1: m is smallest (a=m), need m ∤ (b+c) for all b,c in S with b > m, c >= b
        for b in S_list:
            if b <= m:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % m == 0:
                    return False
        
        # Case 2: m is middle (b=m), need a ∤ (m+c) for all a in S with a < m, c in S with c >= m
        for a in S_list:
            if a >= m:
                continue
            for c in S_list:
                if c < m:
                    continue
            # Only check if c >= m
            for c in S_list:
                if c < m:
                    continue
                if (m + c) % a == 0:
                    return False
        
        # Case 3: m is largest (c=m), need a ∤ (b+m) for all a < b in S with b <= m
        for a in S_list:
            for b in S_list:
                if a >= b:
                    continue
                if b > m:
                    continue
                if (b + m) % a == 0:
                    return False
        
        return True
    
    # Strategy 1: Start with {171,...,256}, add from 170 down to 1
    S1 = set(range(171, n + 1))
    for m in range(170, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_add(S1, m):
            S1.add(m)
    
    best_S = set(S1)
    best_size = len(S1)
    
    # Strategy 2: Start with {171,...,256}, add in order of least constraint
    # Precompute for each candidate m, how many violations it would cause
    S2 = set(range(171, n + 1))
    candidates = list(range(170, 0, -1))
    
    # We'll iteratively pick the candidate that causes the fewest "near misses"
    # For simplicity, just try adding in ascending order (smaller numbers first might be easier to fit)
    S2 = set(range(171, n + 1))
    for m in range(1, 171):
        if time.time() > deadline:
            break
        if is_valid_add(S2, m):
            S2.add(m)
    
    if len(S2) > best_size:
        best_S = set(S2)
        best_size = len(S2)
    
    # Strategy 3: Local search - try to swap elements
    # For each element in best_S, try removing it and adding a smaller element not in S
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_curr = set(best_S)
        in_S = set(S_curr)
        not_in_S = [m for m in range(1, n + 1) if m not in in_S]
        
        for remove_m in list(S_curr):
            if time.time() > deadline:
                break
            S_test = S_curr - {remove_m}
            for add_m in not_in_S:
                if add_m >= remove_m:
                    continue  # Only add smaller elements
                if is_valid_add(S_test, add_m):
                    new_S = S_test | {add_m}
                    # This swap keeps same size, but might enable future improvements
                    # For now, just accept if valid (same size)
                    # Actually, this doesn't improve size. Let's skip pure swaps.
                    pass
        # Try adding without removing
        for add_m in not_in_S:
            if time.time() > deadline:
                break
            if is_valid_add(S_curr, add_m):
                S_curr.add(add_m)
                best_S = set(S_curr)
                best_size = len(S_curr)
                improved = True
                not_in_S = [m for m in range(1, n + 1) if m not in in_S]
                break
    
    # Final verification
    S_final = sorted(best_S)
    valid = True
    for i, a in enumerate(S_final):
        for j in range(i + 1, len(S_final)):
            b = S_final[j]
            for c_val in S_final[j:]:
                c = c_val
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to known valid set
        return list(range(171, 257))
    
    return S_final
