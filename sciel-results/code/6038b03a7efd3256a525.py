def construct():
    import time
    start = time.time()
    DEADLINE = 85.0

    N = 128
    nums = list(range(1, N + 1))
    
    # prime exponent signature per number
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

    # pair_conflict[u][w] = bitmask of v != u,w with {u,v,w} bad
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

    # Greedy initialization
    def greedy(order):
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

    orders = [
        list(range(N)),
        list(range(N - 1, -1, -1)),
        sorted(range(N), key=lambda i: (nums[i].bit_length(), nums[i])),
        sorted(range(N), key=lambda i: -len(sigs[i])),
    ]
    best_init = []
    for o in orders:
        g = greedy(o)
        if len(g) > len(best_init):
            best_init = g
    
    best_mask = 0
    for v in best_init:
        best_mask |= (1 << v)
    best_size = len(best_init)
    
    # Simulated Annealing
    import random
    random.seed(42)
    
    # Precompute list of bits for faster iteration?
    # Or just use bit manipulation.
    
    def popcount(x):
        return x.bit_count()
        
    # Calculate number of bad triples in mask C
    # This is O(N^3) if done naively, but we can do it incrementally or use the pair_conflict structure.
    # For SA, we need delta.
    
    def count_bad_triples(C):
        # C is a bitmask
        # Iterate pairs u < v in C, check if pair_conflict[u][v] & C has any bits
        # Sum of popcounts / 2? No, each bad triple {u,v,w} is counted once for pair (u,v) if w is in mask,
        # once for (u,w) if v is in mask, once for (v,w) if u is in mask.
        # So sum of popcounts over all pairs in C is 3 * N_bad.
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
                # pair_conflict[u][v] contains w such that {u,v,w} bad
                # We only care about w in C.
                # Note: u and v are in C. w is distinct.
                count += popcount(pair_conflict[u][v] & C)
        return count // 3

    # Initial energy
    P = 1000.0
    current_mask = best_mask
    current_size = best_size
    current_bad = count_bad_triples(current_mask)
    current_energy = -current_size + P * current_bad
    
    best_sa_mask = current_mask
    best_sa_size = current_size
    best_sa_energy = current_energy
    
    T = 10.0
    alpha = 0.999
    # We have ~80s left. 
    # Estimate iterations: 
    # Each iter: pick random bit, calculate delta.
    # Delta calc: sum over set bits in C. ~120 bits.
    # ~500k iterations in 1 sec? Maybe less in Python.
    # Let's run until deadline.
    
    while time.time() < DEADLINE:
        # Pick a random bit to flip
        i = random.randint(0, N - 1)
        bit_i = 1 << i
        
        if current_mask & bit_i:
            # Remove i
            # Delta size: -1 -> Energy +1
            # Delta bad: - (bad triples involving i)
            # bad triples involving i = 1/2 * sum_{u in C, u!=i} popcount(pair_conflict[i][u] & C)
            # Note: C includes i. pair_conflict[i][u] does not contain i or u.
            # So (pair_conflict[i][u] & C) counts w in C.
            # Since i is in C, but i is not in pair_conflict[i][u], it's fine.
            
            sum_bad = 0
            tmp = current_mask & ~bit_i
            while tmp:
                lsb_u = tmp & (-tmp)
                u = lsb_u.bit_length() - 1
                tmp ^= lsb_u
                sum_bad += popcount(pair_conflict[i][u] & current_mask)
            
            delta_bad = -sum_bad / 2.0
            delta_energy = 1.0 + P * delta_bad
        else:
            # Add i
            # Delta size: +1 -> Energy -1
            # Delta bad: + (bad triples involving i)
            sum_bad = 0
            tmp = current_mask
            while tmp:
                lsb_u = tmp & (-tmp)
                u = lsb_u.bit_length() - 1
                tmp ^= lsb_u
                # u != i because i not in current_mask
                sum_bad += popcount(pair_conflict[i][u] & current_mask)
            
            delta_bad = sum_bad / 2.0
            delta_energy = -1.0 + P * delta_bad
            
        new_energy = current_energy + delta_energy
        
        if delta_energy < 0 or random.random() < pow(2.718281828, -delta_energy / T):
            # Accept move
            current_mask ^= bit_i
            current_size += 1 if (current_mask & bit_i) else -1
            current_bad += delta_bad
            current_energy = new_energy
            
            if current_bad == 0 and current_size > best_sa_size:
                best_sa_size = current_size
                best_sa_mask = current_mask
                best_sa_energy = current_energy
        else:
            # Reject move
            pass
            
        T *= alpha
        if T < 1e-6:
            T = 1e-6
            
        # Check time more frequently
        if time.time() > DEADLINE:
            break
            
    # Use the best found
    if best_sa_size > best_size:
        best_size = best_sa_size
        best_mask = best_sa_mask
        
    result = [nums[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
