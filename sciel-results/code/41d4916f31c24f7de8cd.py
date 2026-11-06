def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    N = 128
    nums = list(range(1, N + 1))
    
    # Compute prime exponent signatures
    def prime_exp_sig(n):
        d = {}
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                e = 0
                while m % p == 0:
                    m //= p
                    e += 1
                d[p] = e
            p += 1 if p == 2 else 2
        if m > 1:
            d[m] = d.get(m, 0) + 1
        return d

    sigs = [prime_exp_sig(n) for n in nums]
    
    # Compute square-free kernel (radical) for each number
    def square_free_kernel(n):
        sfk = 1
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                sfk *= p
                while m % p == 0:
                    m //= p
            p += 1 if p == 2 else 2
        if m > 1:
            sfk *= m
        return sfk

    kernels = [square_free_kernel(n) for n in nums]
    
    # is_bad: three distinct elements whose three pairwise LCMs are equal
    def is_bad(a, b, c):
        primes = set(sigs[a].keys()) | set(sigs[b].keys()) | set(sigs[c].keys())
        for p in primes:
            ea = sigs[a].get(p, 0)
            eb = sigs[b].get(p, 0)
            ec = sigs[c].get(p, 0)
            mx = max(ea, eb, ec)
            cnt = (1 if ea == mx else 0) + (1 if eb == mx else 0) + (1 if ec == mx else 0)
            if cnt < 2:
                return False
        return True

    # Build conflict graph: edge between i,j if there exists k such that {i,j,k} is bad
    # But we need triples, so use pair_conflict bitmask approach
    pair_conflict = [[0] * N for _ in range(N)]
    for u in range(N):
        for w in range(u + 1, N):
            mask = 0
            for v in range(N):
                if v == u or v == w:
                    continue
                if is_bad(u, v, w):
                    mask |= (1 << v)
            pair_conflict[u][w] = mask
            pair_conflict[w][u] = mask

    # Structural decomposition: group numbers by their "conflict neighborhood"
    # Two numbers i,j are in the same conflict component if they share at least one
    # common "bad triple partner" — i.e., there exists k such that {i,j,k} is bad.
    # Actually, a cleaner decomposition: for each square-free kernel value, 
    # numbers with that kernel form a "chain" where conflicts are more local.
    
    # Better structural approach: identify which numbers can coexist freely.
    # Key observation: if a number is a power of 2, it only conflicts with other
    # powers of 2 and numbers with specific 2-exponent structure.
    
    # Let's use a different structural insight:
    # The condition for {a,b,c} to be bad is that for every prime p dividing any of them,
    # at least two have the max p-exponent. This is equivalent to: 
    # the three numbers' prime exponent vectors have the property that the coordinate-wise max
    # is achieved by at least two of the three in every coordinate.
    
    # Structural partition: group by the set of primes that appear (support of the signature)
    # Numbers with disjoint prime supports can never be in a bad triple together with a third
    # that only has primes from one group... actually this isn't quite right.
    
    # Let me use a cleaner structural decomposition:
    # For each number n, define its "max-exponent pattern" as the tuple of exponents.
    # Two numbers a,b have a "conflict potential" with c if c could complete a bad triple.
    
    # Practical structural approach: use the fact that if we sort by number of prime factors,
    # numbers with few prime factors have fewer conflicts. Build the solution greedily 
    # starting from "easiest" numbers, then use local search to optimize.
    
    # But the directive says to make a STRUCTURAL change. Let me try:
    # Decompose by the largest prime factor. Numbers whose largest prime factor is p
    # form a group. Within each group, conflicts are more constrained.
    
    # Actually, let me try a completely different structural approach:
    # Use the fact that the problem is equivalent to finding a large subset with no
    # "3-wise LCM-equal" configuration. This is a 3-uniform hypergraph independent set problem.
    # We can solve it via branch-and-bound with good pruning, or via maximum satisfiability.
    
    # Let me try: start with ALL numbers, then iteratively remove elements that are in
    # the most bad triples, using a priority queue. This is a greedy hypergraph independent set.
    
    import heapq
    import random
    
    def popcount(x):
        return x.bit_count()
    
    def count_bad_triples(C):
        count = 0
        tmp = C
        while tmp:
            lsb_u = tmp & (-tmp)
            u = lsb_u.bit_length() - 1
            tmp ^= lsb_u
            tmp2 = tmp
            while tmp2:
                lsb_v = tmp2 & (-tmp2)
                v = lsb_v.bit_length() - 1
                tmp2 ^= lsb_v
                count += popcount(pair_conflict[u][v] & C)
        return count // 3
    
    # Strategy 1: Greedy removal — start with all 128, remove the element in the most bad triples
    def greedy_removal():
        current_mask = (1 << N) - 1
        current_bad = count_bad_triples(current_mask)
        
        while current_bad > 0 and time.time() < DEADLINE - 5:
            # Find the element that is in the most bad triples
            best_removal = -1
            best_reduction = 0
            for i in range(N):
                bit_i = 1 << i
                if not (current_mask & bit_i):
                    continue
                # Count bad triples containing i
                sum_bad = 0
                tmp = current_mask & ~bit_i
                while tmp:
                    lsb_u = tmp & (-tmp)
                    u = lsb_u.bit_length() - 1
                    tmp ^= lsb_u
                    sum_bad += popcount(pair_conflict[i][u] & current_mask)
                reduction = sum_bad // 2
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_removal = i
            
            if best_removal < 0 or best_reduction == 0:
                break
            
            current_mask &= ~(1 << best_removal)
            current_bad -= best_reduction
        
        return current_mask, popcount(current_mask)
    
    # Strategy 2: Greedy addition with structural ordering
    # Order numbers by: (number of prime factors, number itself)
    # Numbers with fewer prime factors are "safer" to include
    def greedy_addition_structural():
        # Sort by number of distinct prime factors, then by value
        order = sorted(range(N), key=lambda i: (len(sigs[i]), nums[i]))
        chosen = []
        for v in order:
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if is_bad(chosen[i], chosen[j], v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(v)
        return chosen
    
    # Strategy 3: Local search from multiple initializations
    def local_search_init(init_mask, time_budget):
        current_mask = init_mask
        current_size = popcount(current_mask)
        current_bad = count_bad_triples(current_mask)
        
        deadline = time.time() + time_budget
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                bit_i = 1 << i
                if current_mask & bit_i:
                    # Try removing i
                    sum_bad = 0
                    tmp = current_mask & ~bit_i
                    while tmp:
                        lsb_u = tmp & (-tmp)
                        u = lsb_u.bit_length() - 1
                        tmp ^= lsb_u
                        sum_bad += popcount(pair_conflict[i][u] & current_mask)
                    delta_bad = -sum_bad // 2
                    new_bad = current_bad + delta_bad
                    if new_bad < current_bad:
                        current_mask ^= bit_i
                        current_size -= 1
                        current_bad = new_bad
                        improved = True
                else:
                    # Try adding i
                    sum_bad = 0
                    tmp = current_mask
                    while tmp:
                        lsb_u = tmp & (-tmp)
                        u = lsb_u.bit_length() - 1
                        tmp ^= lsb_u
                        sum_bad += popcount(pair_conflict[i][u] & current_mask)
                    delta_bad = sum_bad // 2
                    new_bad = current_bad + delta_bad
                    if new_bad == 0 and current_bad == 0:
                        current_mask |= bit_i
                        current_size += 1
                        current_bad = new_bad
                        improved = True
                    elif new_bad < current_bad:
                        current_mask |= bit_i
                        current_size += 1
                        current_bad = new_bad
                        improved = True
        
        return current_mask, current_size
    
    # Generate multiple initializations using different strategies
    initializations = []
    
    # Init 1: All numbers, then greedy removal
    if time.time() < DEADLINE - 10:
        rm_mask, rm_size = greedy_removal()
        initializations.append(rm_mask)
    
    # Init 2: Structural greedy addition
    if time.time() < DEADLINE - 10:
        chosen = greedy_addition_structural()
        mask = 0
        for v in chosen:
            mask |= (1 << v)
        initializations.append(mask)
    
    # Init 3: Standard greedy (ascending)
    if time.time() < DEADLINE - 10:
        chosen = []
        for v in range(N):
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if is_bad(chosen[i], chosen[j], v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(v)
        mask = 0
        for v in chosen:
            mask |= (1 << v)
        initializations.append(mask)
    
    # Init 4: Descending greedy
    if time.time() < DEADLINE - 10:
        chosen = []
        for v in range(N - 1, -1, -1):
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if is_bad(chosen[i], chosen[j], v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(v)
        mask = 0
        for v in chosen:
            mask |= (1 << v)
        initializations.append(mask)
    
    # Init 5: Odd numbers first (known good base)
    if time.time() < DEADLINE - 10:
        order = [i for i in range(N) if nums[i] % 2 == 1] + [i for i in range(N) if nums[i] % 2 == 0]
        chosen = []
        for v in order:
            ok = True
            for i in range(len(chosen)):
                for j in range(i + 1, len(chosen)):
                    if is_bad(chosen[i], chosen[j], v):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                chosen.append(v)
        mask = 0
        for v in chosen:
            mask |= (1 << v)
        initializations.append(mask)
    
    # Find the best initialization
    best_mask = 0
    best_size = 0
    for init_mask in initializations:
        size = popcount(init_mask)
        bad = count_bad_triples(init_mask)
        if bad == 0 and size > best_size:
            best_size = size
            best_mask = init_mask
    
    # If no valid initialization, use the largest one and fix it
    if best_size == 0:
        for init_mask in initializations:
            size = popcount(init_mask)
            if size > best_size:
                best_size = size
                best_mask = init_mask
    
    # Local search from the best initialization
    if time.time() < DEADLINE - 5:
        polished_mask, polished_size = local_search_init(best_mask, DEADLINE - 10 - time.time())
        if polished_size > best_size:
            best_size = polished_size
            best_mask = polished_mask
    
    # Try local search from other initializations too
    for init_mask in initializations:
        if time.time() > DEADLINE - 3:
            break
        if init_mask == best_mask:
            continue
        size = popcount(init_mask)
        if size >= best_size:
            polished_mask, polished_size = local_search_init(init_mask, min(5, DEADLINE - 5 - time.time()))
            if polished_size > best_size:
                best_size = polished_size
                best_mask = polished_mask
    
    result = [nums[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
