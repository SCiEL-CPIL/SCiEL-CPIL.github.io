def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    
    N = 256
    numbers = list(range(2, N+1))
    n = len(numbers)
    idx = {v: i for i, v in enumerate(numbers)}
    
    # Precompute for each pair (i,j) with i<j, the bitmask of k values where
    # numbers[k] divides numbers[i]*numbers[j], or numbers[i] divides numbers[k]*numbers[j],
    # or numbers[j] divides numbers[i]*numbers[k]
    # We use bitmasks for fast intersection checks.
    
    # For each pair (b,c), compute which a's conflict with them
    # conflict_mask[i][j] = bitmask of k such that adding k to a set containing i,j creates a violation
    # Actually, let's precompute: for each pair (i,j), the set of k that would conflict
    # k conflicts with pair (i,j) if:
    #   numbers[k] | numbers[i]*numbers[j]  (k divides product of i,j)
    #   numbers[i] | numbers[k]*numbers[j]  (i divides product of k,j)
    #   numbers[j] | numbers[i]*numbers[k]  (j divides product of i,k)
    
    # Use Python integers as bitmasks (n=255 bits fits easily)
    conflict_mask = [[0] * n for _ in range(n)]
    
    for i in range(n):
        vi = numbers[i]
        for j in range(i+1, n):
            vj = numbers[j]
            prod_ij = vi * vj
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                vk = numbers[k]
                if prod_ij % vk == 0:
                    mask |= (1 << k)
                elif (vk * vj) % vi == 0:
                    mask |= (1 << k)
                elif (vi * vk) % vj == 0:
                    mask |= (1 << k)
            conflict_mask[i][j] = mask
            conflict_mask[j][i] = mask
    
    # Greedy: add numbers from largest to smallest
    included_mask = 0  # bitmask of included indices
    included_set = set()
    
    for a in sorted(numbers, reverse=True):
        ia = idx[a]
        # Check if adding ia conflicts with any pair in included_set
        # A conflict exists if there are b,c in included_set (b!=c) such that ia conflicts with pair (b,c)
        # i.e., conflict_mask[b][c] has bit ia set, OR conflict_mask[ia][b] has bit c set, etc.
        # Actually simpler: for each pair (b,c) in included_set, check if ia is in conflict_mask[b][c]
        # But also need to check if ia as part of new pairs conflicts
        # Let me just check: for each b in included_set, for each c in included_set with c!=b,
        # check if ia conflicts with (b,c)
        
        conflict = False
        inc_list = list(included_set)
        for bi in range(len(inc_list)):
            b = inc_list[bi]
            for ci in range(bi+1, len(inc_list)):
                c = inc_list[ci]
                if (conflict_mask[b][c] >> ia) & 1:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            included_mask |= (1 << ia)
            included_set.add(ia)
    
    best_mask = included_mask
    best_size = len(included_set)
    
    # Local search: try swaps to improve
    # For each excluded element, try adding it and removing minimal conflicting elements
    excluded = [i for i in range(n) if not (included_mask >> i) & 1]
    
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        for x in excluded:
            if time.time() > DEADLINE - 5:
                break
            # Check if x can be added by removing some elements
            # Find all elements that conflict with x when added
            # x conflicts with existing element y if there exists z in set such that
            # x conflicts with pair (y,z) or y conflicts with pair (x,z) or z conflicts with pair (x,y)
            # Simpler: try to add x, find all conflicts, remove one, see if size improves
            
            # First check: how many conflicts does x have with current set?
            # For each pair (a,b) in current set, check if x is in conflict_mask[a][b]
            # Also for each a in current set, check if x conflicts with a via some third element
            # This is complex. Let's just try: add x, then remove elements until valid, check if net gain > 0
            
            # Try simple: for each y in included_set, try swap x for y
            # Check if (included - {y} + {x}) is valid
            for y in list(included_set):
                if time.time() > DEADLINE - 5:
                    break
                # Check if removing y and adding x gives a valid set
                # New set = included_set - {y} + {x}
                # Check all triples in new set
                new_set = (included_set - {y}) | {x}
                valid = True
                new_list = list(new_set)
                for ai in range(len(new_list)):
                    a = new_list[ai]
                    for bi in range(ai+1, len(new_list)):
                        b = new_list[bi]
                        for ci in range(bi+1, len(new_list)):
                            c = new_list[ci]
                            va = numbers[a]
                            vb = numbers[b]
                            vc = numbers[c]
                            if (vb * vc) % va == 0 or (va * vc) % vb == 0 or (va * vb) % vc == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    included_set = new_set
                    included_mask = 0
                    for s in included_set:
                        included_mask |= (1 << s)
                    best_mask = included_mask
                    best_size = len(included_set)
                    excluded = [i for i in range(n) if not (included_mask >> i) & 1]
                    improved = True
                    break
    
    result = [numbers[i] for i in range(n) if (best_mask >> i) & 1]
    return sorted(result)
