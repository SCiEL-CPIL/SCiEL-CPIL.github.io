def construct():
    import time
    start_time = time.time()
    deadline = 85.0
    
    N = 256
    S = set(range(129, 257))
    
    def get_violations(S):
        S_list = sorted(S)
        S_set = S
        violations = []
        for a in S_list:
            # b > a, c >= b
            # b+c = k*a
            # k >= 3
            for k in range(3, 513 // a + 1):
                m = k * a
                if m > 512:
                    break
                b_lo = a + 1
                b_hi = m // 2
                for b in range(b_lo, b_hi + 1):
                    c = m - b
                    if b <= c and b in S_set and c in S_set:
                        violations.append((a, b, c))
        return violations

    # Make S valid
    for _ in range(200):
        if time.time() > deadline - 10:
            break
        viols = get_violations(S)
        if not viols:
            break
        counts = {}
        for a, b, c in viols:
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1
            counts[c] = counts.get(c, 0) + 1
        max_count = max(counts.values())
        candidates = [x for x, cnt in counts.items() if cnt == max_count]
        to_remove = max(candidates) # Remove largest to keep density high?
        S.remove(to_remove)

    # Add elements from 128 down to 1
    def is_valid_add(x, S):
        S_set = S
        S_list = sorted(S)
        
        # x as a
        for k in range(3, 513 // x + 1):
            m = k * x
            if m > 512: break
            b_lo = max(x + 1, m - 256)
            b_hi = min(256, m // 2)
            for b in range(b_lo, b_hi + 1):
                c = m - b
                if b <= c and b in S_set and c in S_set:
                    return False
        
        # x as b
        for a in S_list:
            if a >= x: break
            k_lo = (2 * x + a - 1) // a
            k_hi = (256 + x) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - x
                if c >= x and c in S_set:
                    return False
        
        # x as c
        for a in S_list:
            if a >= x: break
            k_lo = (a + x) // a + 1
            k_hi = (2 * x) // a
            for k in range(k_lo, k_hi + 1):
                b = k * a - x
                if a < b <= x and b in S_set:
                    return False
        
        return True

    for x in range(128, 0, -1):
        if time.time() > deadline - 5:
            break
        if x in S:
            continue
        if is_valid_add(x, S):
            S.add(x)
            
    return sorted(S)
