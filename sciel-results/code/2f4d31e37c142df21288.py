def construct():
    import math
    import time
    import heapq

    start_time = time.time()
    deadline = start_time + 115.0

    # Primes up to ~100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    
    # Max-heap for top K candidates based on divisor count (tau)
    # We use negative tau because Python's heapq is a min-heap
    candidates = [] 
    K = 1000
    
    def calc_score(n):
        if n < 2: return 0
        # Generate divisors
        divs = [1]
        temp = n
        i = 0
        # Factorize using the primes list
        while i < len(primes) and primes[i] * primes[i] <= temp:
            p = primes[i]
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            i += 1
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        
        divs.sort()
        
        score = 0
        # Optimization: gcd(a,b) for adjacent divisors
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score

    def dfs(idx, n, max_exp, div_count):
        if time.time() > deadline - 20.0:
            return
            
        if idx >= len(primes):
            # Add to candidates
            if len(candidates) < K:
                heapq.heappush(candidates, (-div_count, n))
            else:
                # If current is better than the worst in heap
                if div_count > -candidates[0][0]:
                    heapq.heapreplace(candidates, (-div_count, n))
            return
        
        p = primes[idx]
        curr = n
        exp = 0
        while curr <= LIMIT:
            exp += 1
            if exp > max_exp:
                break
            dfs(idx + 1, curr, exp, div_count * (exp + 1))
            curr *= p
            if curr > LIMIT:
                break

    # Initialize
    # Max exponent for 2 is 60
    # We can limit max_exp to speed up, but let's allow reasonable exponents
    # For larger primes, max_exp is small
    dfs(0, 1, 60, 1)
    
    # Extract top candidates
    top_candidates = []
    while candidates:
        neg_tau, n = heapq.heappop(candidates)
        top_candidates.append(n)
        
    # Remove duplicates and sort by tau (descending) to prioritize
    # We don't have tau stored in top_candidates, so we just evaluate them.
    # To be safe, we can evaluate the top ones first.
    # Since we popped from a max-heap of -tau, the first ones popped are the largest tau.
    
    best_n = 2
    best_score = 0
    
    # First, evaluate the primorial of first 15 primes as a strong baseline
    primorial = 1
    for p in primes[:15]:
        primorial *= p
    if primorial <= LIMIT:
        s = calc_score(primorial)
        if s > best_score:
            best_score = s
            best_n = primorial
    
    # Evaluate top candidates
    # We might have duplicates, so use a set
    seen = set()
    seen.add(best_n)
    for n in top_candidates:
        if time.time() > deadline - 5.0:
            break
        if n in seen:
            continue
        seen.add(n)
        
        s = calc_score(n)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
