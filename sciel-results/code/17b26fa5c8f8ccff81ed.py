import time
from math import gcd

def construct():
    N = 512
    start = time.time()
    deadline = start + 140.0  # Safety margin under 150s kill

    # Precompute prime factors for each number 1..511
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    pf = [set() for _ in range(N)]
    for i in range(1, N):
        pf[i] = prime_factors(i)

    # Precompute weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)

    def score(S):
        return sum(weights[a] for a in S)

    def is_coprime_with_all(a, S):
        """Check if a is coprime with all elements in S using prime factor sets."""
        fa = pf[a]
        if not fa:
            return True
        for b in S:
            if fa & pf[b]:
                return False
        return True

    def greedy(order):
        S = []
        for a in order:
            if is_coprime_with_all(a, S):
                S.append(a)
        return S

    # Initial greedy in descending order (highest weight first)
    best = greedy(range(N - 1, 0, -1))
    best_score = score(best)
    best_set = set(best)

    # Local search: drop one, re-greedy (this is the key improvement)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for x in list(best):
            if time.time() > deadline - 15:
                break
            # Remove x and re-greedy on remaining candidates
            cand_order = [a for a in range(N - 1, 0, -1) if a != x]
            S2 = greedy(cand_order)
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
                for y in range(N - 1, 0, -1):
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
        for a in range(N - 1, 0, -1):
            if a not in best_set and is_coprime_with_all(a, best):
                best.append(a)
                best_set.add(a)
        best_score = score(best)

    return best
