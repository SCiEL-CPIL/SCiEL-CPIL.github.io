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
            for j in range(i*i, N + 1, i):
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

    # Precompute weight-descending order for greedy restarts
    w_desc = sorted(range(511, 0, -1), key=lambda a: -weights[a])

    # Precompute coprime relationship as a 2D boolean matrix for O(1) lookup
    coprime = [[False] * N for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            coprime[a][b] = (mask[a] & mask[b] == 0)

    # For each a, precompute sorted list of coprime candidates in weight order
    coprime_w_desc = [[] for _ in range(N)]
    for a in range(1, N):
        coprime_w_desc[a] = [b for b in w_desc if coprime[a][b]]

    improved = True
    while improved and time_ok():
        improved = False
        n = len(best)
        best_set = set(best)

        # Move 1: try adding a number not in best (direct add if coprime to all)
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
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 2: swap one element x in best with one not in best
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            best_minus_x_mask = 0
            for b in best:
                if b != x:
                    best_minus_x_mask |= mask[b]
            
            # Find best candidate a not in best that is coprime to all best except x
            best_candidate = -1
            best_candidate_weight = 0.0
            for a in w_desc:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                if mask[a] & best_minus_x_mask == 0:
                    if weights[a] > best_candidate_weight:
                        best_candidate_weight = weights[a]
                        best_candidate = a
                    break  # w_desc is sorted by weight, so first valid is best
            
            if best_candidate >= 0 and best_candidate_weight > weights[x] + 1e-12:
                # Perform swap
                best[i] = best_candidate
                best_set.remove(x)
                best_set.add(best_candidate)
                best_mask = 0
                for b in best:
                    best_mask |= mask[b]
                best_score += best_candidate_weight - weights[x]
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 3: drop two elements and add one (2-for-1 swap)
        for i in range(n):
            if not time_ok():
                break
            for j in range(i + 1, n):
                if not time_ok():
                    break
                x, y = best[i], best[j]
                best_minus_xy_mask = 0
                for b in best:
                    if b != x and b != y:
                        best_minus_xy_mask |= mask[b]
                
                # Find best candidate a not in best that is coprime to all best except x,y
                best_candidate = -1
                best_candidate_weight = 0.0
                for a in w_desc:
                    if not time_ok():
                        break
                    if a in best_set:
                        continue
                    if mask[a] & best_minus_xy_mask == 0:
                        if weights[a] > best_candidate_weight:
                            best_candidate_weight = weights[a]
                            best_candidate = a
                        break
                
                if best_candidate >= 0 and best_candidate_weight > weights[x] + weights[y] + 1e-12:
                    # Remove x and y, add best_candidate
                    best.remove(x)
                    best.remove(y)
                    best.append(best_candidate)
                    best_set.remove(x)
                    best_set.remove(y)
                    best_set.add(best_candidate)
                    best_mask = 0
                    for b in best:
                        best_mask |= mask[b]
                    best_score += best_candidate_weight - weights[x] - weights[y]
                    improved = True
                    break
            if improved:
                break

        if improved or not time_ok():
            continue

        # Move 4: drop one element and add two (1-for-2 swap)
        for i in range(n):
            if not time_ok():
                break
            x = best[i]
            best_minus_x_mask = 0
            for b in best:
                if b != x:
                    best_minus_x_mask |= mask[b]
            
            # Find two candidates a, b not in best that are coprime to all best except x
            # and coprime to each other
            best_pair = None
            best_pair_weight = 0.0
            candidates = []
            for a in w_desc:
                if not time_ok():
                    break
                if a in best_set:
                    continue
                if mask[a] & best_minus_x_mask == 0:
                    candidates.append(a)
                if len(candidates) >= 10:  # Limit search space
                    break
            
            for idx_a in range(len(candidates)):
                if not time_ok():
                    break
                a = candidates[idx_a]
                for idx_b in range(idx_a + 1, len(candidates)):
                    if not time_ok():
                        break
                    b = candidates[idx_b]
                    if mask[a] & mask[b] == 0:
                        pair_weight = weights[a] + weights[b]
                        if pair_weight > best_pair_weight:
                            best_pair_weight = pair_weight
                            best_pair = (a, b)
            
            if best_pair is not None and best_pair_weight > weights[x] + 1e-12:
                a, b = best_pair
                best[i] = a
                best.append(b)
                best_set.remove(x)
                best_set.add(a)
                best_set.add(b)
                best_mask = 0
                for c in best:
                    best_mask |= mask[c]
                best_score += best_pair_weight - weights[x]
                improved = True
                break

        if improved or not time_ok():
            continue

        # Move 5: drop one element and re-greedy on remaining candidates
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

        # Move 6: try adding a number with conflicts, dropping conflicts and re-greedy
        for a in w_desc:
            if not time_ok():
                break
            if a in best_set:
                continue
            # Find conflicts in best
            conflicts = [b for b in best if not coprime[a][b]]
            if not conflicts:
                continue  # Already handled in Move 1
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

        # Move 7: remove two elements, re-greedy
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
