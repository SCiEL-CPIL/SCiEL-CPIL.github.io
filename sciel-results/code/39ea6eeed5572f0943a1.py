def construct():
    import math
    import time
    import random

    start_time = time.time()
    deadline = start_time + 100.0

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18

    def calc_score_fast(n, prime_list):
        if n < 2: return 0
        factors = []
        temp = n
        for i, p in enumerate(prime_list):
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp, 1 << i))
        
        # If temp > 1, it is a prime factor larger than sqrt(n) or not in list.
        # In our construction, all factors are in prime_list.
        
        divs = [(1, 0)]
        for p, exp, bit in factors:
            new_divs = []
            power = 1
            for _ in range(exp + 1):
                for val, mask in divs:
                    new_divs.append((val * power, mask | bit))
                power *= p
            divs = new_divs
            
        divs.sort(key=lambda x: x[0])
        
        score = 0
        for j in range(len(divs) - 1):
            if (divs[j][1] & divs[j+1][1]) == 0:
                score += 1
        return score

    # Find max k for primorial
    prod = 1
    k = 0
    for p in primes:
        if prod * p > LIMIT:
            break
        prod *= p
        k += 1
    
    exponents = [1] * k
    best_n = 1
    for i in range(k):
        best_n *= primes[i]
    best_score = calc_score_fast(best_n, primes)
    
    def n_from_exps(exps):
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
        return n

    # Local Search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # 1. Try adjusting exponents
        for i in range(k):
            for delta in [1, -1]:
                if time.time() > deadline - 10: break
                new_exps = exponents[:]
                new_exps[i] += delta
                if new_exps[i] < 0:
                    continue
                n = n_from_exps(new_exps)
                if n > LIMIT or n < 2:
                    continue
                s = calc_score_fast(n, primes)
                if s > best_score:
                    best_score = s
                    best_n = n
                    exponents = new_exps
                    improved = True
        
        # 2. Random perturbations
        if random.random() < 0.2 and time.time() < deadline - 20:
            new_exps = exponents[:]
            for _ in range(2):
                idx = random.randint(0, k-1)
                delta = random.choice([-2, -1, 1, 2])
                new_exps[idx] += delta
            n = n_from_exps(new_exps)
            if n <= LIMIT and n >= 2:
                s = calc_score_fast(n, primes)
                if s > best_score:
                    best_score = s
                    best_n = n
                    exponents = new_exps
                    improved = True

    return best_n
