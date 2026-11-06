def construct():
    import time
    from math import lcm
    
    deadline = time.time() + 75  # safety margin
    
    def is_valid_triple(a, b, c):
        return not (lcm(a, b) == lcm(a, c) == lcm(b, c))
    
    # Start with all elements
    S = list(range(1, 129))
    removed = []
    
    # Phase 1: Remove elements causing most violations
    while time.time() < deadline:
        violation_count = {x: 0 for x in S}
        found_violation = False
        n = len(S)
        
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                ab = lcm(a, b)
                for k in range(j + 1, n):
                    c = S[k]
                    if ab == lcm(a, c) == lcm(b, c):
                        violation_count[a] += 1
                        violation_count[b] += 1
                        violation_count[c] += 1
                        found_violation = True
        
        if not found_violation:
            break
        
        # Remove element with most violations (tie-break: larger value first to keep small numbers)
        max_count = max(violation_count[x] for x in S)
        candidates = [x for x in S if violation_count[x] == max_count]
        max_elem = max(candidates)  # remove largest to preserve smaller numbers
        S.remove(max_elem)
        removed.append(max_elem)
    
    # Phase 2: Try to add back removed elements
    for x in removed:
        if time.time() > deadline - 10:
            break
        # Check if adding x back creates any violation
        can_add = True
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                if lcm(a, b) == lcm(a, x) == lcm(b, x):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            S.append(x)
    
    if not S:
        S = [1]
    
    return S
