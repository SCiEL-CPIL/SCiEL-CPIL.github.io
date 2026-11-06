def construct():
    import time
    import random
    import math
    
    deadline = time.time() + 115.0
    N = 511

    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    pindex = {p: i for i, p in enumerate(primes)}

    mult_mask = {}
    for p in primes:
        m = 0
        for b in range(p, N + 1, p):
            m |= (1 << b)
        mult_mask[p] = m

    pmask = [0] * (N + 1)
    conflict = [0] * (N + 1)
    for a in range(1, N + 1):
        if a == 1:
            pmask[a] = 0
            conflict[a] = 0
            continue
        x = a
        pm = 0
        cm = 0
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                pm |= (1 << pindex[p])
                cm |= mult_mask[p]
                while x % p == 0:
                    x //= p
        if x > 1:
            pm |= (1 << pindex[x])
            cm |= mult_mask[x]
        pmask[a] = pm
        conflict[a] = cm

    w = [0.0] * (N + 1)
    for a in range(1, N + 1):
        w[a] = 1.0 / (512 - a)

    # Precompute all valid elements sorted by weight descending
    all_elements = list(range(1, N + 1))
    all_elements.sort(key=lambda a: -w[a])
    
    # Precompute for each element, the list of elements it conflicts with
    # and the list of elements it does NOT conflict with (compatible)
    compatible = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a != b and (conflict[a] & (1 << b)) == 0:
                compatible[a].append(b)
    
    def greedy_init(seed_offset=0):
        """Greedy initialization with diversity through different scoring."""
        S = set()
        S_mask = 0
        rng = random.Random(42 + seed_offset)
        
        if seed_offset == 0:
            # Pure greedy: highest weight first
            elements = all_elements[:]
        elif seed_offset == 1:
            # Weighted by w[a] / (1 + num_conflicts) - opportunity cost
            def opp_cost(a):
                num_conf = sum(1 for b in all_elements if b != a and (conflict[a] & (1 << b)) != 0)
                return w[a] / (1 + num_conf * 0.1)
            elements = sorted(all_elements, key=lambda a: -opp_cost(a))
        else:
            # Random perturbation of greedy
            elements = all_elements[:]
            half = len(elements) // 2
            rng.shuffle(elements[half:])
        
        for a in elements:
            if (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
        return S, S_mask

    def polish(S, S_mask):
        """Add any remaining non-conflicting elements in weight order."""
        for a in all_elements:
            if not (S_mask >> a) & 1 and (conflict[a] & S_mask) == 0:
                S_mask |= (1 << a)
                S.add(a)
        return S, S_mask

    def local_search(S, S_mask, current_score, rng, max_iter, deadline):
        """Simulated annealing local search with precomputed compatibility."""
        T = 0.08
        T_min = 1e-8
        cooling = 0.9995
        best_score = current_score
        best_S = S.copy()
        best_S_mask = S_mask
        
        # Track which elements are in S for fast membership
        in_S = [False] * (N + 1)
        for a in S:
            in_S[a] = True
        
        for iteration in range(max_iter):
            if time.time() >= deadline - 3.0:
                break
            if T < T_min:
                break
                
            move_type = rng.random()
            delta = 0.0
            new_S_mask = S_mask
            added = None
            removed = None
            
            if move_type < 0.40:
                # Try to add a random non-conflicting element
                candidates = [a for a in all_elements if not in_S[a] and (conflict[a] & S_mask) == 0]
                if candidates:
                    a = rng.choice(candidates)
                    new_S_mask |= (1 << a)
                    delta = w[a]
                    added = a
            elif move_type < 0.60:
                # Try to remove a random element
                if S:
                    x = rng.choice(list(S))
                    new_S_mask ^= (1 << x)
                    delta = -w[x]
                    removed = x
            else:
                # Try to swap: remove one, add another
                if S:
                    x = rng.choice(list(S))
                    rest = S_mask ^ (1 << x)
                    candidates = [a for a in all_elements if not in_S[a] and a != x and (conflict[a] & rest) == 0]
                    if candidates:
                        a = rng.choice(candidates)
                        new_S_mask = rest | (1 << a)
                        delta = w[a] - w[x]
                        added = a
                        removed = x
            
            if added is None and removed is None:
                T *= cooling
                continue
            
            # Acceptance criterion
            if delta > 0 or (delta / T) > -700 and rng.random() < math.exp(delta / T):
                if added is not None:
                    S.add(added)
                    in_S[added] = True
                if removed is not None:
                    S.discard(removed)
                    in_S[removed] = False
                S_mask = new_S_mask
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = S.copy()
                    best_S_mask = S_mask
            
            T *= cooling
        
        return best_S, best_S_mask, best_score

    # Multi-start simulated annealing
    global_best_S = None
    global_best_S_mask = 0
    global_best_score = -1.0
    
    num_starts = 10
    rng = random.Random(12345)
    
    for start_idx in range(num_starts):
        if time.time() >= deadline - 5.0:
            break
        
        S, S_mask = greedy_init(start_idx)
        S, S_mask = polish(S, S_mask)
        current_score = sum(w[a] for a in S)
        
        if current_score > global_best_score:
            global_best_score = current_score
            global_best_S = S.copy()
            global_best_S_mask = S_mask
        
        # Run local search
        S, S_mask, score = local_search(S, S_mask, current_score, rng, 50000, deadline)
        
        if score > global_best_score:
            global_best_score = score
            global_best_S = S.copy()
            global_best_S_mask = S_mask
    
    # Final polish on the best solution found
    final_S, final_S_mask = polish(global_best_S.copy(), global_best_S_mask)
    final_score = sum(w[a] for a in final_S)
    
    if final_score > global_best_score:
        global_best_S = final_S
        global_best_S_mask = final_S_mask
        global_best_score = final_score
    
    return list(global_best_S)
