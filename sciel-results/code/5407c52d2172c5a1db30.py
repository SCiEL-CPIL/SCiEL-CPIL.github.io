def construct():
    import time
    from math import gcd

    N = 512
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s kill

    def score(S):
        return sum(1.0 / (N - a) for a in S)

    def is_coprime_with_all(a, S):
        for b in S:
            if gcd(a, b) != 1:
                return False
        return True

    def greedy(order):
        S = []
        for a in order:
            if is_coprime_with_all(a, S):
                S.append(a)
        return S

    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in range(1, 512)}

    # Multi-start with different orderings
    orderings = [
        range(511, 0, -1),  # descending (highest weight first)
        # Sort by weight but with slight perturbation for diversity
        sorted(range(1, 512), key=lambda a: -weights[a] + 0.001 * (a % 7)),
        sorted(range(1, 512), key=lambda a: -weights[a] + 0.001 * (a % 11)),
        # Reverse for diversity
        range(1, 512),
    ]

    best = None
    best_score = -1.0
    best_set = set()

    for ordering in orderings:
        if time.time() > deadline - 20:
            break
        cand = greedy(ordering)
        s = score(cand)
        if s > best_score:
            best = cand
            best_score = s
            best_set = set(cand)

    # Local search: drop one, re-greedy
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for x in list(best):
            if time.time() > deadline - 15:
                break
            cand = [a for a in range(511, 0, -1) if a != x]
            S2 = greedy(cand)
            s2 = score(S2)
            if s2 > best_score + 1e-12:
                best = S2
                best_score = s2
                best_set = set(best)
                improved = True
                break

    # Pair swap: remove one, add another (must be coprime with all remaining)
    if time.time() < deadline - 10:
        improved2 = True
        while improved2 and time.time() < deadline - 10:
            improved2 = False
            for x in list(best):
                if time.time() > deadline - 10:
                    break
                remaining = [a for a in best if a != x]
                # Try candidates in descending weight order
                for y in range(511, 0, -1):
                    if y == x or y in best_set:
                        continue
                    if is_coprime_with_all(y, remaining):
                        S2 = remaining + [y]
                        s2 = score(S2)
                        if s2 > best_score + 1e-12:
                            best = S2
                            best_score = s2
                            best_set = set(best)
                            improved2 = True
                            break
                if improved2:
                    break

    # Final greedy pass to fill in any gaps
    if time.time() < deadline - 2:
        best_set = set(best)
        for a in range(511, 0, -1):
            if a not in best_set and is_coprime_with_all(a, best):
                best.append(a)
                best_set.add(a)
        best_score = score(best)

    return best
