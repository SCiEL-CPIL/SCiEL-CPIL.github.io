def construct():
    import time
    from math import gcd
    N = 512
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150s kill

    # Precompute conflict masks: mask[a] has bit b set if gcd(a, b) != 1
    # Index by value 1..511, so we use a dict or offset list
    # Use a list indexed 0..511 (index 0 unused)
    mask = [0] * 512
    for a in range(1, 512):
        m = 0
        for b in range(1, 512):
            if a != b and gcd(a, b) != 1:
                m |= (1 << b)
        mask[a] = m

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def greedy(order):
        S = []
        chosen_mask = 0
        for a in order:
            if (chosen_mask & mask[a]) == 0:
                S.append(a)
                chosen_mask |= (1 << a)
        return S

    # Initial greedy in descending order (prefers large a -> large 1/(512-a))
    best = greedy(range(511, 0, -1))
    best_score = score(best)

    # Local search: drop one chosen element, then re-greedy descending.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in list(best):
            cand = [a for a in range(511, 0, -1) if a != x]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                improved = True
                break

    # Second phase: try swapping one element out and re-greedy
    # Also try different orderings to escape local optima
    if time.time() < deadline:
        # Try ascending + descending hybrid orders
        orders = [
            range(511, 0, -1),
            sorted(range(1, 512), key=lambda a: -1.0/(N-a)),
        ]
        for order in orders:
            if time.time() >= deadline:
                break
            S_new = greedy(order)
            s_new = score(S_new)
            if s_new > best_score + 1e-12:
                best = S_new
                best_score = s_new

        # Swap-based local search: for each pair (in_best, not_in_best),
        # try removing in_best and adding not_in_best if coprime to rest
        improved2 = True
        while improved2 and time.time() < deadline:
            improved2 = False
            best_set = set(best)
            best_mask = 0
            for a in best:
                best_mask |= (1 << a)
            for x in list(best):
                if time.time() >= deadline:
                    break
                # Remove x, see what greedy would add
                # Try adding each candidate not in best that is coprime to (best - {x})
                rest_mask = best_mask & ~(1 << x)
                best_without_x = [a for a in best if a != x]
                best_without_x_score = sum(1.0 / (N - a) for a in best_without_x)
                for y in range(511, 0, -1):
                    if y in best_set:
                        continue
                    if (rest_mask & mask[y]) != 0:
                        continue
                    # y is coprime to all in best_without_x
                    new_score = best_without_x_score + 1.0 / (N - y)
                    if new_score > best_score + 1e-12:
                        best = best_without_x + [y]
                        best_score = new_score
                        best_set = set(best)
                        best_mask = 0
                        for a in best:
                            best_mask |= (1 << a)
                        improved2 = True
                        break

    return best
