def construct():
    import time
    deadline = time.time() + 110  # safety margin for 120s limit
    
    def find_primes_mod4(limit):
        """Find primes congruent to 1 mod 4 up to limit."""
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [p for p in range(5, limit + 1) if sieve[p] and p % 4 == 1]
    
    primes_1mod4 = find_primes_mod4(5000)
    # Use first ~50 smallest primes ≡ 1 mod 4
    primes = primes_1mod4[:50]
    
    best_product = 1
    best_exponents = []
    
    def dfs(idx, remaining_log, current_product, exponents):
        nonlocal best_product, best_exponents
        if time.time() > deadline - 5:
            return
        if idx >= len(primes):
            if current_product > best_product:
                best_product = current_product
                best_exponents = exponents[:]
            return
        p = primes[idx]
        # Try exponents from 0 up to max possible
        max_e = 0
        while max_e + 1 <= 60 and current_product * (p ** (max_e + 1)) <= 10**18:
            max_e += 1
        # Greedy: try highest exponents first for early primes
        for e in range(max_e, -1, -1):
            new_product = current_product * (p ** e)
            if new_product > 10**18:
                break
            exponents.append(e)
            dfs(idx + 1, remaining_log, new_product, exponents)
            exponents.pop()
            # Pruning: if current product * (p^(max_e+1)) > 10^18 already handled
            if time.time() > deadline - 5:
                return
    
    dfs(0, 0, 1, [])
    
    # Compute N and its representations as sum of two squares
    N = 1
    for i, e in enumerate(best_exponents):
        if e > 0:
            N *= primes[i] ** e
    
    if N > 10**18:
        N = 10**18
        # Fallback: use a simpler construction
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Find all representations x^2 + y^2 = N using Gaussian integer arithmetic
    # For N = prod p_i^e_i with p_i ≡ 1 mod 4, each p_i = a_i^2 + b_i^2
    # We need to find (a,b) for each prime, then combine
    
    def sum_of_squares_prime(p):
        """Find a, b such that a^2 + b^2 = p for p ≡ 1 mod 4."""
        for a in range(1, int(p**0.5) + 1):
            b2 = p - a*a
            b = int(b2**0.5)
            if b*b == b2:
                return a, b
        return None
    
    # Get representations for each prime power
    # p^e has e+1 representations (up to order and sign)
    # Representations of p: (a, b)
    # p^2: (a^2-b^2, 2ab) and (a^2+b^2, 0) -- but we want positive
    # General: use Brahmagupta-Fibonacci identity recursively
    
    def rep_prime_power(p, e):
        """Generate all representations (x,y) with x>=y>0 of p^e as sum of two squares."""
        base = sum_of_squares_prime(p)
        if base is None:
            return []
        a0, b0 = base
        if a0 < b0:
            a0, b0 = b0, a0
        # p^1: [(a0, b0)]
        # Use complex multiplication: (a0+b0i)^k for k=0..e, then take real/imag
        # Actually we need all ways, which correspond to distributing the e factors
        # among conjugate pairs
        
        # Simpler: enumerate all (x,y) such that x^2 + y^2 = p^e
        # Using the fact that each representation corresponds to choosing
        # how many "copies" of the prime go to each Gaussian factor
        # Number of primitive-ish reps: e+1
        # All reps including imprimitive: need more care
        
        # Direct enumeration for small e
        reps = set()
        pe = p ** e
        for x in range(1, int(pe**0.5) + 1):
            y2 = pe - x*x
            if y2 <= 0:
                break
            y = int(y2**0.5)
            if y*y == y2:
                reps.add((max(x,y), min(x,y)))
        return sorted(reps, reverse=True)
    
    # For the full N, combine representations using identity
    def combine_reps(reps1, reps2):
        """Combine (a^2+b^2)(c^2+d^2) = (ac-bd)^2+(ad+bc)^2"""
        result = set()
        for a, b in reps1:
            for c, d in reps2:
                x1 = a*c - b*d
                y1 = a*d + b*c
                x2 = a*c + b*d
                y2 = a*d - b*c
                for x, y in [(abs(x1), abs(y1)), (abs(x2), abs(y2))]:
                    if x > 0 and y > 0:
                        result.add((max(x,y), min(x,y)))
        return result
    
    # Build up representations for N
    if not any(best_exponents):
        return {'N': 2, 'pairs': [[1, 1]]}
    
    current_reps = [(1, 0)]  # Representation of 1
    for i, e in enumerate(best_exponents):
        if e == 0:
            continue
        p = primes[i]
        p_reps = rep_prime_power(p, e)
        if not p_reps:
            continue
        if not current_reps or current_reps == [(1,0)]:
            current_reps = p_reps
        else:
            current_reps = list(combine_reps(current_reps, p_reps))
        if time.time() > deadline - 10:
            break
    
    # Filter: need x^2 + y^2 = N exactly, and x <= y
    valid_pairs = []
    for x, y in current_reps:
        if x*x + y*y == N and x <= y and x > 0:
            valid_pairs.append([x*x, y*y])
    
    # Deduplicate
    valid_pairs = list(set(tuple(p) for p in valid_pairs))
    valid_pairs.sort()
    
    if not valid_pairs:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': N, 'pairs': valid_pairs}
