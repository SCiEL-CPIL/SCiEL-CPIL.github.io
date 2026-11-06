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

    P = 1000.0
    best_sa_mask = best_mask
    best_sa_size = best_size
    
    def run_sa(init_mask, T0, alpha, seed, time_budget):
        random.seed(seed)
        current_mask = init_mask
        current_size = popcount(current_mask)
        current_bad = count_bad_triples(current_mask)
        current_energy = -current_size + P * current_bad
        
        local_best_mask = current_mask
        local_best_size = current_size
        local_best_energy = current_energy
        
        T = T0
        deadline = time.time() + time_budget
        
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            bit_i = 1 << i
            
            if current_mask & bit_i:
                # Remove i
                sum_bad = 0
                tmp = current_mask & ~bit_i
                while tmp:
                    lsb_u = tmp & (-tmp)
                    u = lsb_u.bit_length() - 1
                    tmp ^= lsb_u
                    sum_bad += popcount(pair_conflict[i][u] & current_mask)
                
                delta_bad = -sum_bad // 2
                delta_energy = 1.0 + P * delta_bad
            else:
                # Add i
                sum_bad = 0
                tmp = current_mask
                while tmp:
                    lsb_u = tmp & (-tmp)
                    u = lsb_u.bit_length() - 1
                    tmp ^= lsb_u
                    sum_bad += popcount(pair_conflict[i][u] & current_mask)
                
                delta_bad = sum_bad // 2
                delta_energy = -1.0 + P * delta_bad
            
            new_energy = current_energy + delta_energy
            
            if delta_energy < 0 or random.random() < pow(2.718281828, -delta_energy / T):
                current_mask ^= bit_i
                current_size += 1 if (current_mask & bit_i) else -1
                current_bad += delta_bad
                current_energy = new_energy
                
                if current_bad == 0 and current_size > local_best_size:
                    local_best_size = current_size
                    local_best_mask = current_mask
                    local_best_energy = current_energy
            else:
                pass
                
            T *= alpha
            if T < 1e-6:
                T = 1e-6
        
        return local_best_mask, local_best_size

    # Run multiple SA restarts with different parameters
    sa_configs = [
        (best_mask, 15.0, 0.9995, 42, 15),
        (best_mask, 8.0, 0.999, 123, 15),
        (best_mask, 25.0, 0.9998, 456, 15),
        (best_mask, 12.0, 0.9992, 789, 15),
        (best_mask, 5.0, 0.998, 101, 15),
    ]
    
    for config in sa_configs:
        if time.time() > DEADLINE - 5:
            break
        init_mask, T0, alpha, seed, budget = config
        if time.time() + budget > DEADLINE - 2:
            budget = max(1, DEADLINE - 2 - time.time())
        sa_mask, sa_size = run_sa(init_mask, T0, alpha, seed, budget)
        if sa_size > best_sa_size:
            best_sa_size = sa_size
            best_sa_mask = sa_mask
    
    # Final local search polish: try all single-bit flips and accept improving moves
    def local_polish(mask, max_rounds=50):
        current_mask = mask
        current_size = popcount(current_mask)
        current_bad = count_bad_triples(current_mask)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < DEADLINE - 1:
            improved = False
            rounds += 1
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
                    new_size = current_size - 1
                    new_bad = current_bad + delta_bad
                    # Accept if it reduces bad triples, or maintains bad=0 and we're not losing size
                    if (new_bad < current_bad) or (new_bad == 0 and current_bad == 0):
                        # Only accept removal if it helps or is neutral when bad=0
                        if new_bad < current_bad:
                            current_mask ^= bit_i
                            current_size = new_size
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
                    new_size = current_size + 1
                    new_bad = current_bad + delta_bad
                    if new_bad == 0 and current_bad == 0:
                        current_mask |= bit_i
                        current_size = new_size
                        current_bad = new_bad
                        improved = True
                    elif new_bad < current_bad:
                        current_mask |= bit_i
                        current_size = new_size
                        current_bad = new_bad
                        improved = True
        return current_mask, current_size

    if best_sa_size > best_size:
        best_size = best_sa_size
        best_mask = best_sa_mask
    
    # Polish the best mask
    polished_mask, polished_size = local_polish(best_mask)
    if polished_size > best_size:
        best_size = polished_size
        best_mask = polished_mask
    
    result = [nums[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
