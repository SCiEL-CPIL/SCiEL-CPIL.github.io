def construct():
    from math import gcd
    import time
    
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    N = 512
    # weights: weight[a] = 1/(512 - a) for a in 1..511
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # Precompute gcd matrix or just use gcd on the fly
    # For speed, precompute prime factors
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
    
    factors = {a: prime_factors(a) for a in range(1, N)}
    factors[1] = set()  # 1 has no prime factors
    
    def is_coprime(a, b):
        if a == 1 or b == 1:
            return True
        return len(factors[a] & factors[b]) == 0
    
    # Greedy: sort by weight descending
    candidates = sorted(range(1, N), key=lambda a: -weights[a])
    chosen = []
    for a in candidates:
        if all(is_coprime(a, b) for b in chosen):
            chosen.append(a)
    
    def score(chosen):
        return sum(weights[a] for a in chosen)
    
    best = chosen[:]
    best_score = score(best)
    
    # Local search: try removing one and re-greedy
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, a in enumerate(best):
            if time.time() > deadline:
                break
            # Remove a
            remaining = [x for x in best if x != a]
            # Greedily add back the best available
            new_chosen = remaining[:]
            for b in candidates:
                if b in new_chosen:
                    continue
                if all(is_coprime(b, c) for c in new_chosen):
                    new_chosen.append(b)
                    if time.time() > deadline:
                        break
            new_score = score(new_chosen)
            if new_score > best_score + 1e-12:
                best = new_chosen
                best_score = new_score
                improved = True
    
    return best
