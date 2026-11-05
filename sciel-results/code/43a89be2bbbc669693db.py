def construct():
    import math
    import time
    import heapq
    import random

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
        if primorial > LIMIT:
            primorial //= p
            break
    if primorial > 1:
        s = calc_score(primorial)
        if s > best_score:
            best_score = s
            best_n = primorial

    # Evaluate top candidates
    # We might have duplicates, so use a set
    seen = set()
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
            
    # Local Search / Polish Phase
    # Try to improve the best candidate found so far
    # We will perform a few iterations of local search on the top few candidates
    
    def local_search(n, max_iters=20):
        if time.time() > deadline - 2.0:
            return n, 0
            
        current_n = n
        current_score = calc_score(current_n)
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
                
            improved = False
            # Factorize current_n to get primes and exponents
            temp = current_n
            factors = []
            i = 0
            while i < len(primes) and primes[i] * primes[i] <= temp:
                p = primes[i]
                if temp % p == 0:
                    e = 0
                    while temp % p == 0:
                        temp //= p
                        e += 1
                    factors.append((p, e))
                i += 1
            if temp > 1:
                factors.append((temp, 1))
                
            # Try modifications
            new_n = current_n
            new_score = current_score
            
            # 1. Try increasing exponent of a prime
            for p, e in factors:
                if current_n * p <= LIMIT:
                    candidate_n = current_n * p
                    if candidate_n not in seen:
                        seen.add(candidate_n)
                        s = calc_score(candidate_n)
                        if s > new_score:
                            new_score = s
                            new_n = candidate_n
                            improved = True
                            
            # 2. Try decreasing exponent of a prime
            for p, e in factors:
                if e > 1:
                    candidate_n = current_n // p
                    if candidate_n not in seen:
                        seen.add(candidate_n)
                        s = calc_score(candidate_n)
                        if s > new_score:
                            new_score = s
                            new_n = candidate_n
                            improved = True
                elif e == 1:
                    # Try removing the prime entirely
                    candidate_n = current_n // p
                    if candidate_n > 1 and candidate_n not in seen:
                        seen.add(candidate_n)
                        s = calc_score(candidate_n)
                        if s > new_score:
                            new_score = s
                            new_n = candidate_n
                            improved = True

            # 3. Try swapping a prime with a nearby prime
            # This is more complex, so we'll stick to exponent changes for now to save time
            # But we can try replacing a prime p with p' where p' is close to p
            for idx, (p, e) in enumerate(factors):
                # Look for nearby primes in the global list
                # Find index of p in primes
                if p in primes:
                    pi = primes.index(p)
                    # Check neighbors
                    for offset in [-2, -1, 1, 2]:
                        ni = pi + offset
                        if 0 <= ni < len(primes):
                            p_new = primes[ni]
                            if p_new == p: continue
                            # Calculate new n: remove p^e, add p_new^e
                            # n_new = n / p^e * p_new^e
                            if current_n % (p**e) == 0:
                                base = current_n // (p**e)
                                candidate_n = base * (p_new**e)
                                if 1 < candidate_n <= LIMIT and candidate_n not in seen:
                                    seen.add(candidate_n)
                                    s = calc_score(candidate_n)
                                    if s > new_score:
                                        new_score = s
                                        new_n = candidate_n
                                        improved = True

            if improved:
                current_n = new_n
                current_score = new_score
            else:
                break
                
        return current_n, current_score

    # Apply local search to the current best and a few other top candidates
    # We need to re-evaluate top candidates to get a list of (score, n)
    # But we already have best_n. Let's just polish best_n.
    # Also, we should polish the primorial if it was good.
    
    # Let's polish the top 5 unique candidates from the heap
    # We need to retrieve them again or store them.
    # Since we popped them, we have `top_candidates`.
    # Let's re-score the first 10 unique ones and polish the best among them.
    
    polished_candidates = []
    for n in top_candidates[:20]: # Look at top 20 from heap
        if n in seen:
            continue
        seen.add(n)
        s = calc_score(n)
        polished_candidates.append((s, n))
        
    polished_candidates.sort(reverse=True)
    
    for s, n in polished_candidates[:5]:
        if time.time() > deadline - 5.0:
            break
        new_n, new_s = local_search(n)
        if new_s > best_score:
            best_score = new_s
            best_n = new_n

    return best_n
