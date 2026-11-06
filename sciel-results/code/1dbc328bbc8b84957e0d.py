import time
import bisect

def construct():
    start = time.time()
    deadline = start + 82.0
    n = 256
    
    def full_validate(S_list):
        n2 = len(S_list)
        for i in range(n2):
            a = S_list[i]
            for j in range(i+1, n2):
                b = S_list[j]
                for k in range(j, n2):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add(a, S_set, S_sorted):
        """Check if a can be added to S_set (all elements > a)."""
        # For each multiple m of a in [2(a+1), 2n], check if m = b+c for b,c in S, b<=c
        for m in range(2*(a+1), 2*n + 1, a):
            b_min = a + 1
            b_max = m // 2
            # Use bisect to find elements in S_sorted within [b_min, b_max]
            left = bisect.bisect_left(S_sorted, b_min)
            right = bisect.bisect_right(S_sorted, b_max)
            for idx in range(left, right):
                b = S_sorted[idx]
                c = m - b
                if c >= b and c in S_set:
                    return False
        return True
    
    # Start with the known good base: [171, 256]
    best_S = set(range(171, n+1))
    best_size = len(best_S)
    
    # Try various starting thresholds
    for m_start in [165, 160, 155, 150, 145, 140, 135, 130, 128, 120, 110, 100]:
        if time.time() > deadline - 8:
            break
        
        # Check if [m_start, n] is valid
        base = list(range(m_start, n+1))
        valid = True
        for i in range(len(base)):
            a = base[i]
            for j in range(i+1, len(base)):
                b = base[j]
                for k in range(j, len(base)):
                    c = base[k]
                    if (b + c) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        S_list = base[:]
        S_set = set(S_list)
        
        if len(S_list) > best_size:
            best_S = set(S_list)
            best_size = len(S_list)
        
        # Add elements below m_start, trying from largest to smallest
        for a in range(m_start - 1, 0, -1):
            if time.time() > deadline - 3:
                break
            if a in S_set:
                continue
            if can_add(a, S_set, S_list):
                pos = bisect.bisect_left(S_list, a)
                S_list.insert(pos, a)
                S_set.add(a)
                if len(S_list) > best_size:
                    best_S = set(S_list)
                    best_size = len(S_list)
    
    # Final polish: try to add any remaining elements to best_S
    if time.time() < deadline - 15:
        S_list = sorted(best_S)
        S_set = set(S_list)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            # Try adding elements in descending order (larger first)
            for a in range(n, 0, -1):
                if a in S_set:
                    continue
                if can_add(a, S_set, S_list):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    changed = True
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)
        
        # Second pass: try ascending order to find elements missed
        if time.time() < deadline - 5:
            S_list = sorted(best_S)
            S_set = set(S_list)
            for a in range(1, n + 1):
                if a in S_set:
                    continue
                if can_add(a, S_set, S_list):
                    pos = bisect.bisect_left(S_list, a)
                    S_list.insert(pos, a)
                    S_set.add(a)
                    if len(S_list) > best_size:
                        best_S = set(S_list)
                        best_size = len(S_list)
    
    # Final validation
    if time.time() < deadline:
        S_final = sorted(best_S)
        if len(S_final) > 5 and not full_validate(S_final):
            S_final = list(range(171, n+1))
        return S_final
    
    return sorted(best_S)
