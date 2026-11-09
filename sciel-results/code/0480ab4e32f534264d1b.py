import math
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = 210
    max_val = 10210

    best_answer = None
    best_raw = float('inf')

    def update_best(ans, raw):
        nonlocal best_answer, best_raw
        if raw < best_raw:
            prod = 210
            for x in ans:
                prod *= x
            r = math.isqrt(prod)
            if r * r == prod:
                if all(isinstance(x, int) and 210 < x <= max_val for x in ans):
                    if len(set(ans)) == len(ans):
                        best_answer = [int(x) for x in ans]
                        best_raw = raw

    # 1. Single element: x = 210 * k^2 > 210 => k >= 2
    for k in range(2, 100):
        x = 210 * k * k
        if x > max_val:
            break
        update_best([x], x - 210)

    # 2. Two elements: x * y = 210 * k^2, x != y, x, y > 210, x, y <= max_val
    for k in range(1, 200):
        if time.time() > deadline - 30:
            break
        val = 210 * k * k
        if val > max_val * max_val:
            break
        sqrt_val = math.isqrt(val)
        lo = max(211, math.ceil(val / max_val))
        hi = min(sqrt_val, max_val)
        for x in range(lo, hi + 1):
            if val % x == 0:
                y = val // x
                if y > 210 and y <= max_val and x != y:
                    m = max(x, y)
                    if m - 210 < best_raw:
                        update_best([x, y], m - 210)

    # Precompute parity sets efficiently using smallest prime factor sieve
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def prime_parity_set(n):
        s = set()
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                s.add(p)
        return frozenset(s)

    target_parity = frozenset({2, 3, 5, 7})

    # 3. Three elements using parity-set approach
    # Precompute parity for [211, 10210]
    parity_range = min(10210, 210 + int(best_raw) if best_raw < float('inf') else 10210)
    # We want to search up to best_raw - 1 to try to improve
    search_max = min(10210, 210 + max(1, int(best_raw) - 1) if best_raw < float('inf') else 10210)
    parity_range = max(search_max, 230)
    
    parity_cache = {}
    for i in range(211, parity_range + 1):
        parity_cache[i] = prime_parity_set(i)
    
    # Group numbers by parity set
    parity_groups = defaultdict(list)
    for i in range(211, parity_range + 1):
        parity_groups[parity_cache[i]].append(i)
    
    # For 3 elements: need px XOR py XOR pz = target
    # For each z, for each x < z, compute required parity for y:
    #   py = target XOR px XOR pz
    #   Check if there exists y in (x, z) with that parity
    
    found3 = False
    for z in range(211, parity_range + 1):
        if time.time() > deadline - 10:
            break
        if z - 210 >= best_raw:
            break
        
        pz = parity_cache[z]
        
        for x in range(211, z):
            px = parity_cache[x]
            required_py = target_parity.symmetric_difference(px).symmetric_difference(pz)
            
            if required_py in parity_groups:
                lst = parity_groups[required_py]
                lo, hi = 0, len(lst)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if lst[mid] <= x:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < len(lst) and lst[lo] < z:
                    y = lst[lo]
                    m = z
                    if m - 210 < best_raw:
                        update_best([x, y, z], m - 210)
                        found3 = True
                        break
        
        if found3:
            break

    # 4. If no 3-element solution or best_raw is still large, try 4 elements
    if time.time() < deadline - 15 and best_raw > 50:
        M4 = min(210 + max(1, int(best_raw) - 1), 500)
        if M4 < 211:
            M4 = 230
        
        pairs_by_parity4 = defaultdict(list)
        
        for x in range(211, M4 + 1):
            px = parity_cache[x] if x in parity_cache else prime_parity_set(x)
            for y in range(x + 1, M4 + 1):
                py = parity_cache[y] if y in parity_cache else prime_parity_set(y)
                pxy = px.symmetric_difference(py)
                m_xy = max(x, y)
                pairs_by_parity4[pxy].append((m_xy, x, y))
        
        found4 = False
        for p1, pair_list1 in pairs_by_parity4.items():
            if time.time() > deadline - 5:
                break
            p2 = target_parity.symmetric_difference(p1)
            if p2 in pairs_by_parity4:
                for m1, a, b in pair_list1:
                    if time.time() > deadline - 5:
                        break
                    if m1 - 210 >= best_raw:
                        continue
                    for m2, c, d in pairs_by_parity4[p2]:
                        if a != c and a != d and b != c and b != d:
                            m = max(m1, m2)
                            if m - 210 < best_raw:
                                update_best([a, b, c, d], m - 210)
                                found4 = True
                                break
                    if found4:
                        break
            if found4:
                break

    # 5. Systematic small-max search: try all subsets of {211, ..., 210+best_raw-1}
    if time.time() < deadline - 5:
        max_search = min(10210, 210 + max(1, int(best_raw) - 1))
        candidates = list(range(211, max_search + 1))
        
        # Try 5-element combinations (more elements give more flexibility)
        from itertools import combinations
        
        # Precompute parities for all candidates
        cand_parities = {}
        for c in candidates:
            if c in parity_cache:
                cand_parities[c] = parity_cache[c]
            else:
                cand_parities[c] = prime_parity_set(c)
        
        # For 5 elements: need XOR of all 5 parities = target
        # Try all 5-combinations
        found5 = False
        for combo in combinations(candidates, 5):
            if time.time() > deadline - 2:
                break
            m = max(combo)
            if m - 210 >= best_raw:
                continue
            xor_p = frozenset()
            for c in combo:
                xor_p = xor_p.symmetric_difference(cand_parities[c])
            if xor_p == target_parity:
                update_best(list(combo), m - 210)
                found5 = True
                break
        
        # If still no improvement, try 4-element combinations
        if not found5 and time.time() < deadline - 2:
            found4b = False
            for combo in combinations(candidates, 4):
                if time.time() > deadline - 1:
                    break
                m = max(combo)
                if m - 210 >= best_raw:
                    continue
                xor_p = frozenset()
                for c in combo:
                    xor_p = xor_p.symmetric_difference(cand_parities[c])
                if xor_p == target_parity:
                    update_best(list(combo), m - 210)
                    found4b = True
                    break
    
    if best_answer:
        return best_answer
    
    return [840]
