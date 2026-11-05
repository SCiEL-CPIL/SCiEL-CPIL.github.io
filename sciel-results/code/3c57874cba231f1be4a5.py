def construct():
    import time
    from math import gcd
    
    start_time = time.time()
    deadline = 110  # safety margin before 120s
    
    N = 512
    
    # Precompute weights
    weights = {a: 1.0 / (N - a) for a in range(1, N)}
    
    # Precompute prime factors for each number
    prime_factors = {}
    for a in range(1, N):
        factors = set()
        x = a
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        prime_factors[a] = factors
    
    # Greedy initialization
    chosen = []
    used_primes = set()
    for a in range(N - 1, 0, -1):
        if prime_factors[a].isdisjoint(used_primes):
            chosen.append(a)
            used_primes.update(prime_factors[a])
    
    def get_score(chosen_set):
        return sum(weights[a] for a in chosen_set)
    
    best_score = get_score(chosen)
    
    # Local search: try single swaps
    chosen_set = set(chosen)
    
    def is_valid_add(chosen_set, a):
        return prime_factors[a].isdisjoint(p for p in prime_factors.get(a, set()))  # wrong
    
    # Better: check if a is coprime to all in chosen_set
    def can_add(chosen_set, a):
        for b in chosen_set:
            if gcd(a, b) != 1:
                return False
        return True
    
    # Actually, using prime factors is faster
    def get_used_primes(chosen_set):
        s = set()
        for a in chosen_set:
            s.update(prime_factors[a])
        return s
    
    used = get_used_primes(chosen_set)
    
    # Local search with swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_swap = None
        best_gain = 0
        current_score = get_score(chosen_set)
        
        # Try removing one and adding one or more
        for a_remove in list(chosen_set):
            new_used = used - prime_factors[a_remove]
            # Try adding candidates not in chosen_set
            for a_add in range(N - 1, 0, -1):
                if a_add in chosen_set:
                    continue
                if not prime_factors[a_add].isdisjoint(new_used):
                    continue
                gain = weights[a_add] - weights[a_remove]
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_swap = (a_remove, a_add)
                    improved = True
        
        if best_swap:
            a_remove, a_add = best_swap
            chosen_set.discard(a_remove)
            chosen_set.add(a_add)
            used = get_used_primes(chosen_set)
    
    return list(chosen_set)
