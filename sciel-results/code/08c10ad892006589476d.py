def construct():
    import time
    from math import gcd
    from fractions import Fraction
    from itertools import combinations
    import random
    
    N = 512
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150s kill

    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(512)]
    for p in range(2, 512):
        for m in range(p, 512, p):
            prime_factors[m].add(p)

    # Precompute conflict masks using prime factor overlap
    mask = [0] * 512
    for a in range(1, 512):
        if not prime_factors[a]:
            continue
        m = 0
        for pf in prime_factors[a]:
            for b in range(pf, 512, pf):
                if b != a:
                    m |= (1 << b)
        mask[a] = m

    def score_float(S):
        return sum(1.0 / (N - a) for a in S)

    def score_frac(S):
        return sum(Fraction(1, N - a) for a in S)

    def greedy(order, chosen_mask=0):
        S = []
        cm = chosen_mask
        for a in order:
            if (cm & mask[a]) == 0:
                S.append(a)
                cm |= (1 << a)
        return S, cm

    # Initial greedy in descending order
    best, best_mask = greedy(range(511, 0, -1))
    best_score = score_float(best)
    best_frac = score_frac(best)

    def is_valid_set(S):
        used_primes = set()
        for a in S:
            for pf in prime_factors[a]:
                if pf in used_primes:
                    return False
                used_primes.add(pf)
        return True

    # Local search with multiple move types
    def local_search(S, smask, time_limit):
        S = list(S)
        smask = smask
        s_score = score_float(S)
        s_frac = score_frac(S)
        improved = True
        
        while improved and time.time() < time_limit:
            improved = False
            
            # Move type 1: Drop one element and re-greedy
            for x in list(S):
                if time.time() >= time_limit:
                    break
                order = [a for a in range(511, 0, -1) if a != x]
                S2, _ = greedy(order)
                s2 = score_float(S2)
                if s2 > s_score + 1e-12:
                    S = S2
                    smask = 0
                    for a in S:
                        smask |= (1 << a)
                    s_score = s2
                    s_frac = score_frac(S)
                    improved = True
                    break
            
            # Move type 2: 1-for-1 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            for x in list(S):
                if time.time() >= time_limit:
                    break
                rest_mask = smask & ~(1 << x)
                S_without_x = [a for a in S if a != x]
                best_y = None
                best_y_gain = None
                for y in range(511, 0, -1):
                    if y in S_set:
                        continue
                    if (rest_mask & mask[y]) != 0:
                        continue
                    gain = Fraction(1, N - y) - Fraction(1, N - x)
                    if best_y_gain is None or gain > best_y_gain:
                        best_y_gain = gain
                        best_y = y
                if best_y is not None and best_y_gain > 0:
                    new_S = S_without_x + [best_y]
                    new_score = score_float(new_S)
                    if new_score > s_score + 1e-12:
                        S = new_S
                        smask = 0
                        for a in S:
                            smask |= (1 << a)
                        s_score = new_score
                        s_frac = score_frac(S)
                        S_set = set(S)
                        improved = True
                        break
            
            # Move type 3: 2-for-1 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            n_S = len(S)
            for i in range(n_S):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_S):
                    if time.time() >= time_limit:
                        break
                    x, y = S[i], S[j]
                    rest_mask = smask & ~(1 << x) & ~(1 << y)
                    S_without_xy = [a for a in S if a != x and a != y]
                    best_z = None
                    best_z_gain = None
                    for z in range(511, 0, -1):
                        if z in S_set:
                            continue
                        if (rest_mask & mask[z]) != 0:
                            continue
                        gain = Fraction(1, N - z) - Fraction(1, N - x) - Fraction(1, N - y)
                        if best_z_gain is None or gain > best_z_gain:
                            best_z_gain = gain
                            best_z = z
                    if best_z is not None and best_z_gain > 0:
                        new_S = S_without_xy + [best_z]
                        new_score = score_float(new_S)
                        if new_score > s_score + 1e-12:
                            S = new_S
                            smask = 0
                            for a in S:
                                smask |= (1 << a)
                            s_score = new_score
                            s_frac = score_frac(S)
                            S_set = set(S)
                            n_S = len(S)
                            improved = True
                            break
                else:
                    continue
                break
            
            # Move type 4: 2-for-2 swap
            if time.time() >= time_limit:
                break
            S_set = set(S)
            n_S = len(S)
            for i in range(n_S):
                if time.time() >= time_limit:
                    break
                for j in range(i + 1, n_S):
                    if time.time() >= time_limit:
                        break
                    x, y = S[i], S[j]
                    rest_mask = smask & ~(1 << x) & ~(1 << y)
                    S_without_xy = [a for a in S if a != x and a != y]
                    
                    # Find best pair (z, w) to add
                    best_pair = None
                    best_pair_gain = None
                    candidates = [z for z in range(511, 0, -1) if z not in S_set and (rest_mask & mask[z]) == 0]
                    for idx1 in range(len(candidates)):
                        if time.time() >= time_limit:
                            break
                        z = candidates[idx1]
                        temp_mask = rest_mask | (1 << z)
                        for idx2 in range(idx1 + 1, len(candidates)):
                            if time.time() >= time_limit:
                                break
                            w = candidates[idx2]
                            if (temp_mask & mask[w]) != 0:
                                continue
                            gain = Fraction(1, N - z) + Fraction(1, N - w) - Fraction(1, N - x) - Fraction(1, N - y)
                            if best_pair_gain is None or gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (z, w)
                    
                    if best_pair is not None and best_pair_gain > 0:
                        z, w = best_pair
                        new_S = S_without_xy + [z, w]
                        new_score = score_float(new_S)
                        if new_score > s_score + 1e-12:
                            S = new_S
                            smask = 0
                            for a in S:
                                smask |= (1 << a)
                            s_score = new_score
                            s_frac = score_frac(S)
                            S_set = set(S)
                            n_S = len(S)
                            improved = True
                            break
                else:
                    continue
                break
            
            # Final polish: try all single additions
            if time.time() >= time_limit:
                break
            S_set = set(S)
            for z in range(511, 0, -1):
                if z in S_set:
                    continue
                if (smask & mask[z]) != 0:
                    continue
                new_S = S + [z]
                new_score = score_float(new_S)
                if new_score > s_score + 1e-12:
                    S = new_S
                    smask |= (1 << z)
                    s_score = new_score
                    s_frac = score_frac(S)
                    S_set = set(S)
                    improved = True
        
        return S, smask, s_score

    # Run local search on the initial solution
    best, best_mask, best_score = local_search(best, best_mask, deadline)

    # Try randomized restarts with different orderings
    random.seed(42)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        max_restarts = int(remaining_time / 8.0)  # Budget ~8s per restart
        for restart in range(max_restarts):
            if time.time() >= deadline:
                break
            # Create a perturbed order
            order = list(range(1, 512))
            r = random.random()
            if r < 0.3:
                # Mostly descending with small perturbations
                order = list(range(511, 0, -1))
                for _ in range(random.randint(1, 10)):
                    i = random.randint(0, len(order) - 2)
                    j = random.randint(i + 1, min(i + 50, len(order) - 1))
                    order[i], order[j] = order[j], order[i]
            elif r < 0.6:
                # Random shuffle
                random.shuffle(order)
            else:
                # Sort by random key weighted by 1/(512-a)
                weight_float = {a: 1.0 / (N - a) for a in range(1, 512)}
                keys = [random.random() + 0.1 * weight_float[a] for a in order]
                order = [a for _, a in sorted(zip(keys, order), reverse=True)]
            
            S2, cm2 = greedy(order)
            time_left = deadline - time.time()
            if time_left > 5:
                S2, cm2, s2 = local_search(S2, cm2, min(deadline, time.time() + min(8.0, time_left)))
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = cm2
                    best_score = s2

    return best
