def construct():
    import time
    from math import gcd
    N = 512
    start = time.time()
    deadline = start + 145.0

    # Precompute prime factors as bitmask for each number 1..511
    primes = []
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, N + 1, i):
                is_prime[j] = False

    prime_index = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)

    # For each a in 1..511, compute bitmask of prime factors
    mask = [0] * (N + 1)
    for a in range(2, N):
        m = a
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                mask[a] |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            mask[a] |= (1 << prime_index[m])
    # mask[1] = 0

    weights = [0.0] * (N + 1)
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    def score(S):
        return sum(weights[a] for a in S)

    def greedy(order):
        S = []
        mask_acc = 0
        for a in order:
            if mask[a] & mask_acc == 0:
                S.append(a)
                mask_acc |= mask[a]
        return S

    # Initial greedy descending
    best = greedy(range(511, 0, -1))
    best_mask = 0
    for a in best:
        best_mask |= mask[a]
    best_score = score(best)

    def time_ok():
        return time.time() < deadline

    # Precompute weight-descending order
    w_desc = sorted(range(511, 0, -1), key=lambda a: -weights[a])

    improved = True
    while improved and time_ok():
        improved = False
        n = len(best)
        best_set = set(best)

        # Move 1: drop one element, then re-greedy on remaining candidates
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            cand_order = [a for a in w_desc if a != x]
            S2 = greedy(cand_order)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a in best:
                    best_mask |= mask[a]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 2: try adding a number not in best, and dropping all conflicts
        for a in w_desc:
            if not time_ok():
                break
            if a in best_set:
                continue
            # Find conflicts in best
            conflicts = [b for b in best if mask[a] & mask[b] != 0]
            if not conflicts:
                # Can add directly
                S2 = best + [a]
                s2 = best_score + weights[a]
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break
                continue
            # Try dropping conflicts and adding a, then re-greedy
            drop_set = set(conflicts)
            remaining = [b for b in best if b not in drop_set]
            remaining_set = set(remaining)
            cand_order = [a] + [c for c in w_desc if c != a and c not in remaining_set and c not in drop_set]
            S2 = greedy(cand_order)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_mask = 0
                for a2 in best:
                    best_mask |= mask[a2]
                best_score = s2
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 3: swap one element x in best with one not in best
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            for a in w_desc:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                # Check if a can replace x: a must be coprime to all in best except x
                ok = True
                for b in best:
                    if b != x and mask[a] & mask[b] != 0:
                        ok = False
                        break
                if ok:
                    S2 = [b for b in best if b != x] + [a]
                    s2 = best_score - weights[x] + weights[a]
                    if s2 > best_score + 1e-12:
                        best = S2
                        best_mask = 0
                        for a2 in best:
                            best_mask |= mask[a2]
                        best_score = s2
                        improved = True
                        break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 4: remove two elements, re-greedy
        for i in range(n):
            if not time_ok():
                break
            for j in range(i + 1, n):
                if not time_ok():
                    break
                x, y = best[i], best[j]
                removed = {x, y}
                cand_order = [a for a in w_desc if a not in removed]
                S2 = greedy(cand_order)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 5: multi-swap - remove 2, add up to 3, re-greedy on freed slots
        # This explores configurations where 2 removals allow 3 additions
        for i in range(n):
            if not time_ok():
                break
            for j in range(i + 1, n):
                if not time_ok():
                    break
                x, y = best[i], best[j]
                removed = {x, y}
                # Try adding the top 3 highest-weight numbers not in best and not removed
                add_candidates = [a for a in w_desc[:30] if a not in best_set and a not in removed][:3]
                if not add_candidates:
                    continue
                # Build order: add_candidates first, then rest of w_desc
                add_set = set(add_candidates)
                cand_order = add_candidates + [a for a in w_desc if a not in removed and a not in add_set]
                S2 = greedy(cand_order)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 6: random restart with perturbation - pick random subset to keep, re-greedy
        import random
        if n > 4:
            for _ in range(3):
                if not time_ok():
                    break
                # Keep 70% of current best, re-greedy
                keep_count = max(2, int(n * 0.7))
                keep_indices = random.sample(range(n), keep_count)
                keep_set = {best[i] for i in keep_indices}
                # Build order: kept elements first (in weight order), then rest
                kept_ordered = sorted(keep_set, key=lambda a: -weights[a])
                cand_order = kept_ordered + [a for a in w_desc if a not in keep_set]
                S2 = greedy(cand_order)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True
                    break

        if improved or not time_ok():
            continue

        # No improvement found in all moves - try a full re-greedy from scratch
        S2 = greedy(w_desc)
        s2 = score(S2)
        if s2 > best_score + 1e-12:
            best = S2
            best_mask = 0
            for a in best:
                best_mask |= mask[a]
            best_score = s2
            improved = True
        else:
            # Stuck - try removing the lowest-weight element and re-greedy
            if n > 1:
                min_idx = min(range(n), key=lambda i: weights[best[i]])
                x = best[min_idx]
                cand_order = [a for a in w_desc if a != x]
                S2 = greedy(cand_order)
                s2 = score(S2)
                if s2 > best_score + 1e-12:
                    best = S2
                    best_mask = 0
                    for a in best:
                        best_mask |= mask[a]
                    best_score = s2
                    improved = True

    # Final polish: try adding any remaining coprime number
    best_set = set(best)
    for a in w_desc:
        if not time_ok():
            break
        if a in best_set:
            continue
        if mask[a] & best_mask == 0:
            best.append(a)
            best_mask |= mask[a]
            best_score += weights[a]
            best_set.add(a)

    return best
