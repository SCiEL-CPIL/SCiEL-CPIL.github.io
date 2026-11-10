def construct():
    import time
    from math import gcd
    from fractions import Fraction
    import random

    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s

    # Sieve primes below 512
    def _sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(n+1) if is_prime[i]]

    primes = _sieve(511)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    n_primes = len(primes)

    # Precompute prime factor bitmasks and exact weights
    def prime_factor_mask(n):
        mask = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                mask |= (1 << prime_to_idx[d])
                n //= d
            d += 1
        if n > 1:
            mask |= (1 << prime_to_idx[n])
        return mask

    pfmask = {}
    for a in range(1, 512):
        pfmask[a] = prime_factor_mask(a)

    # Exact weights as Fractions
    weight_frac = {a: Fraction(1, 512 - a) for a in range(1, 512)}
    # Float weights for quick comparison
    weight_float = {a: float(weight_frac[a]) for a in range(1, 512)}

    def exact_score(chosen):
        s = Fraction(0, 1)
        for a in chosen:
            s += weight_frac[a]
        return s

    def float_score(chosen):
        return sum(weight_float[a] for a in chosen)

    def can_add(a, used_mask):
        if a == 1:
            return True
        return (pfmask[a] & used_mask) == 0

    def greedy_select(order=None):
        if order is None:
            order = sorted(range(1, 512), key=lambda x: weight_float[x], reverse=True)
        chosen = []
        used_mask = 0
        chosen_set = set()
        for a in order:
            if a in chosen_set:
                continue
            if a == 1:
                chosen.append(a)
                chosen_set.add(a)
                continue
            if can_add(a, used_mask):
                chosen.append(a)
                chosen_set.add(a)
                used_mask |= pfmask[a]
        return chosen

    # Initial greedy
    best_set = greedy_select()
    best_exact = exact_score(best_set)
    best_float = float_score(best_set)

    # Local search: 1-for-1 swaps using exact arithmetic
    def local_search_1for1(chosen, max_iters=20000):
        chosen_set = set(chosen)
        used_mask = 0
        for a in chosen:
            used_mask |= pfmask[a]

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if (time.time() - start_time) >= DEADLINE:
                break
            improved = False
            iters += 1

            for i, a in enumerate(list(chosen)):
                if a == 1:
                    continue
                # Remove a
                new_used_mask = 0
                for b in chosen:
                    if b != a:
                        new_used_mask |= pfmask[b]

                best_candidate = None
                best_gain = Fraction(0, 1)

                for c in range(511, 0, -1):
                    if c == a:
                        continue
                    if c in chosen_set:
                        continue
                    if c == 1:
                        if 1 in chosen_set:
                            continue
                        gain = weight_frac[1] - weight_frac[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = c
                        continue
                    if (pfmask[c] & new_used_mask) != 0:
                        continue
                    gain = weight_frac[c] - weight_frac[a]
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = c

                if best_candidate is not None:
                    chosen.remove(a)
                    chosen.append(best_candidate)
                    chosen_set = set(chosen)
                    improved = True
                    # Recompute used_mask
                    used_mask = 0
                    for b in chosen:
                        used_mask |= pfmask[b]
                    return True
        return improved

    # 2-for-1 swap: remove two elements, add one
    def local_search_2for1(chosen, max_iters=5000):
        chosen_set = set(chosen)
        n = len(chosen)

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if (time.time() - start_time) >= DEADLINE:
                break
            improved = False
            iters += 1

            for i in range(n):
                a1 = chosen[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, n):
                    a2 = chosen[j]
                    if a2 == 1:
                        continue

                    # Remove both a1 and a2
                    new_used_mask = 0
                    for b in chosen:
                        if b != a1 and b != a2:
                            new_used_mask |= pfmask[b]

                    best_candidate = None
                    best_gain = Fraction(0, 1)

                    for c in range(511, 0, -1):
                        if c == a1 or c == a2:
                            continue
                        if c in chosen_set:
                            continue
                        if c == 1:
                            if 1 in chosen_set:
                                continue
                            gain = weight_frac[1] - weight_frac[a1] - weight_frac[a2]
                            if gain > best_gain:
                                best_gain = gain
                                best_candidate = c
                            continue
                        if (pfmask[c] & new_used_mask) != 0:
                            continue
                        gain = weight_frac[c] - weight_frac[a1] - weight_frac[a2]
                        if gain > best_gain:
                            best_gain = gain
                            best_candidate = c

                    if best_candidate is not None:
                        chosen.remove(a1)
                        chosen.remove(a2)
                        chosen.append(best_candidate)
                        chosen_set = set(chosen)
                        improved = True
                        return True
        return improved

    # 1-for-2 swap: remove one element, add two
    def local_search_1for2(chosen, max_iters=5000):
        chosen_set = set(chosen)

        improved = True
        iters = 0
        while improved and iters < max_iters:
            if (time.time() - start_time) >= DEADLINE:
                break
            improved = False
            iters += 1

            for i, a in enumerate(list(chosen)):
                if a == 1:
                    continue

                # Remove a
                new_used_mask = 0
                for b in chosen:
                    if b != a:
                        new_used_mask |= pfmask[b]

                # Find best pair (c1, c2) to add
                best_pair = None
                best_gain = Fraction(0, 1)

                candidates = []
                for c in range(511, 0, -1):
                    if c == a:
                        continue
                    if c in chosen_set:
                        continue
                    if c == 1:
                        if 1 in chosen_set:
                            continue
                        candidates.append(c)
                    else:
                        if (pfmask[c] & new_used_mask) != 0:
                            continue
                        candidates.append(c)

                # Try pairs from candidates
                # Limit to top candidates for speed
                top_cands = candidates[:80]

                for idx1, c1 in enumerate(top_cands):
                    mask1 = pfmask[c1] if c1 != 1 else 0
                    for c2 in top_cands[idx1+1:]:
                        if c2 == c1:
                            continue
                        if (pfmask[c2] & mask1) != 0:
                            continue
                        gain = weight_frac[c1] + weight_frac[c2] - weight_frac[a]
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (c1, c2)

                if best_pair is not None:
                    chosen.remove(a)
                    chosen.append(best_pair[0])
                    chosen.append(best_pair[1])
                    chosen_set = set(chosen)
                    improved = True
                    return True
        return improved

    # Run local searches
    local_search_1for1(best_set, max_iters=50000)
    new_score = exact_score(best_set)
    if new_score > best_exact:
        best_exact = new_score
        best_float = float(best_exact)

    if (time.time() - start_time) < DEADLINE:
        local_search_2for1(best_set, max_iters=10000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score
            best_float = float(best_exact)

    if (time.time() - start_time) < DEADLINE:
        local_search_1for2(best_set, max_iters=10000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score
            best_float = float(best_exact)

    # More 1-for-1 passes
    if (time.time() - start_time) < DEADLINE:
        local_search_1for1(best_set, max_iters=50000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score
            best_float = float(best_exact)

    # Multiple restarts with different orderings
    for restart in range(30):
        if (time.time() - start_time) >= DEADLINE:
            break

        random.seed(restart * 7919 + 13)

        # Create a perturbed ordering
        base_order = sorted(range(1, 512), key=lambda x: weight_float[x], reverse=True)

        if restart == 0:
            order = base_order
        else:
            # Perturb: shuffle a random subset
            order = list(base_order)
            k = min(20 + restart, len(order))
            shuffle_start = random.randint(0, len(order) - k)
            segment = order[shuffle_start:shuffle_start+k]
            random.shuffle(segment)
            order[shuffle_start:shuffle_start+k] = segment

        chosen = greedy_select(order)
        score = exact_score(chosen)

        if score > best_exact:
            best_exact = score
            best_float = float(score)
            best_set = list(chosen)

        # Local search on this
        if (time.time() - start_time) < DEADLINE:
            local_search_1for1(chosen, max_iters=20000)
            score = exact_score(chosen)
            if score > best_exact:
                best_exact = score
                best_float = float(score)
                best_set = list(chosen)

        if (time.time() - start_time) < DEADLINE:
            local_search_2for1(chosen, max_iters=5000)
            score = exact_score(chosen)
            if score > best_exact:
                best_exact = score
                best_float = float(score)
                best_set = list(chosen)

    # Final polishing passes
    if (time.time() - start_time) < DEADLINE:
        local_search_1for1(best_set, max_iters=50000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score

    if (time.time() - start_time) < DEADLINE:
        local_search_2for1(best_set, max_iters=10000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score

    if (time.time() - start_time) < DEADLINE:
        local_search_1for2(best_set, max_iters=5000)
        new_score = exact_score(best_set)
        if new_score > best_exact:
            best_exact = new_score

    return list(best_set)
