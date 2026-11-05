import math
import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    LIMIT = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def factorize(n):
        """Factorize n using the primes list, return list of (p, exp)"""
        temp = n
        factors = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    def generate_divisors(factors):
        """Generate all divisors from factorization"""
        divs = [1]
        for p, exp in factors:
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs.extend(new_divs)
        divs.sort()
        return divs
    
    def calc_score(n):
        """Calculate tau_perp(n): number of adjacent coprime pairs in sorted divisors"""
        if n < 2:
            return 0
        factors = factorize(n)
        divs = generate_divisors(factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    def estimate_score(n):
        """Quick estimate: use divisor count as proxy for potential score"""
        factors = factorize(n)
        tau = 1
        for p, exp in factors:
            tau *= (exp + 1)
        return tau
    
    # Phase 1: Generate candidates with high divisor count using DFS
    # Keep top K candidates by estimated score (divisor count)
    candidates = []
    K = 2000  # Keep top 2000 candidates
    
    def dfs(idx, n, max_exp_allowed, current_tau):
        if time.time() > deadline - 30.0:
            return
        if idx >= len(primes):
            if len(candidates) < K:
                heapq.heappush(candidates, (-current_tau, n))
            else:
                if current_tau > -candidates[0][0]:
                    heapq.heapreplace(candidates, (-current_tau, n))
            return
        
        p = primes[idx]
        curr = n
        exp = 0
        while curr <= LIMIT:
            exp += 1
            if exp > max_exp_allowed:
                break
            dfs(idx + 1, curr, exp, current_tau * (exp + 1))
            curr *= p
            if curr > LIMIT:
                break
    
    # Start DFS - limit max exponent to keep search manageable
    # For prime 2, max exponent is around 60 (2^60 ~ 10^18)
    dfs(0, 1, 60, 1)
    
    # Extract top candidates
    top_candidates = []
    while candidates:
        neg_tau, n = heapq.heappop(candidates)
        top_candidates.append(n)
    
    # Phase 2: Score candidates and find best
    best_n = 2
    best_score = 0
    seen = set()
    
    # Evaluate primorial of first 15 primes as a strong baseline
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
        seen.add(primorial)
    
    # Score top candidates
    scored_candidates = []
    for n in top_candidates:
        if time.time() > deadline - 15.0:
            break
        if n in seen:
            continue
        seen.add(n)
        s = calc_score(n)
        scored_candidates.append((s, n))
        if s > best_score:
            best_score = s
            best_n = n
    
    # Sort by score descending
    scored_candidates.sort(reverse=True)
    
    # Phase 3: Local search on top candidates
    def local_search(n, max_iters=20):
        """Local search: try increasing/decreasing exponents and swapping primes"""
        if time.time() > deadline - 5.0:
            return n, 0
        
        current_n = n
        current_score = calc_score(current_n)
        
        for iteration in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            factors = factorize(current_n)
            best_neighbor_n = current_n
            best_neighbor_score = current_score
            improved = False
            
            # Try increasing each prime's exponent
            for p, e in factors:
                if current_n * p <= LIMIT:
                    cand = current_n * p
                    if cand not in seen:
                        seen.add(cand)
                        s = calc_score(cand)
                        if s > best_neighbor_score:
                            best_neighbor_score = s
                            best_neighbor_n = cand
                            improved = True
            
            # Try decreasing each prime's exponent
            for p, e in factors:
                if e > 0:
                    cand = current_n // p
                    if cand >= 2 and cand not in seen:
                        seen.add(cand)
                        s = calc_score(cand)
                        if s > best_neighbor_score:
                            best_neighbor_score = s
                            best_neighbor_n = cand
                            improved = True
            
            # Try swapping primes (replace p with adjacent prime q)
            for p, e in factors:
                if p in primes:
                    pi = primes.index(p)
                    for offset in [-1, 1, -2, 2]:
                        ni = pi + offset
                        if 0 <= ni < len(primes):
                            p_new = primes[ni]
                            if p_new == p:
                                continue
                            base = current_n // (p ** e)
                            cand = base * (p_new ** e)
                            if 1 < cand <= LIMIT and cand not in seen:
                                seen.add(cand)
                                s = calc_score(cand)
                                if s > best_neighbor_score:
                                    best_neighbor_score = s
                                    best_neighbor_n = cand
                                    improved = True
            
            if improved:
                current_n = best_neighbor_n
                current_score = best_neighbor_score
            else:
                break
        
        return current_n, current_score
    
    # Apply local search to top candidates
    for s, n in scored_candidates[:15]:
        if time.time() > deadline - 5.0:
            break
        new_n, new_s = local_search(n)
        if new_s > best_score:
            best_score = new_s
            best_n = new_n
    
    return best_n
