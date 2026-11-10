def construct():
    import math
    import time
    from itertools import combinations
    
    deadline = time.time() + 140  # safety margin before 150s limit
    
    def compute_tau_perp(n):
        """Compute tau_perp: number of adjacent coprime pairs in sorted divisors."""
        factors = []
        temp = n
        i = 2
        while i * i <= temp:
            if temp % i == 0:
                exp = 0
                while temp % i == 0:
                    temp //= i
                    exp += 1
                factors.append((i, exp))
            i += 1
        if temp > 1:
            factors.append((temp, 1))
        
        divisors = [1]
        for p, exp in factors:
            new_divs = []
            for d in divisors:
                val = d
                for e in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        
        divisors.sort()
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i+1]) == 1:
                score += 1
        return score
    
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i+2) == 0:
                return False
            i += 6
        return True
    
    # Generate primes up to 200
    candidates = [p for p in range(2, 200) if is_prime(p)]
    
    best_n = 1
    best_score = -1
    
    # Start with first 15 primes
    base_primes = candidates[:15]
    prod = 1
    for p in base_primes:
        prod *= p
    best_n = prod
    best_score = compute_tau_perp(best_n)
    
    # Local search: try swapping one prime at a time
    current = list(base_primes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current)):
            if time.time() > deadline:
                break
            orig = current[idx]
            for new_p in candidates:
                if new_p == orig:
                    continue
                if new_p in current:
                    continue
                test = list(current)
                test[idx] = new_p
                test.sort()
                prod = 1
                for p in test:
                    prod *= p
                    if prod > 10**18:
                        break
                if prod > 10**18:
                    continue
                score = compute_tau_perp(prod)
                if score > best_score:
                    best_score = score
                    best_n = prod
                    current = test
                    improved = True
                    break
    
    return best_n
