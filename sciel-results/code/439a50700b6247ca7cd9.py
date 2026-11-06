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
        # Hybrid approach:
        # For small a, use residue counting which is O(|S| + a)
        # For large a, use multiple checking which is O((n/a) log |S|)
        
        if a < 64:
            # Residue method
            # We need to check if there exist b, c in S such that (b+c) % a == 0
            # Let R[r] be count of x in S with x % a == r
            R = [0] * a
            for x in S_sorted:
                R[x % a] += 1
            
            for r1 in range(a):
                if R[r1] == 0:
                    continue
                r2 = (-r1) % a
                if r1 == r2:
                    # Need 2*r1 % a == 0, which is true by definition of r2=r1
                    # We need at least one element with residue r1 (b=c is allowed)
                    if R[r1] >= 1:
                        return False
                else:
                    # Need one element with r1 and one with r2
                    if R[r2] >= 1:
                        return False
            return True
        else:
            # Original method
            for m in range(2*(a+1), 2*n + 1, a):
                b_min = a + 1
                b_max = m // 2
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
