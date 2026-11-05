import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Precompute primes
    def get_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # We need primes up to roughly 1000 for swapping, maybe a bit more.
    # Product of first 15 primes is ~6.1e17.
    # If we swap 2 with p, p < 3.2.
    # If we swap 3 with p, p < 4.8.
    # If we swap 47 with p, p < 76.
    # If we remove a prime, we can add a larger one.
    # Let's sieve up to 10000 to be safe for potential larger swaps if we drop small primes.
    primes = get_primes(10000)
    prime_set = set(primes)

    # 2. Scoring function
    # Optimized to be as fast as possible.
    def calc_score(n):
        # Factorize n
        factors = []
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
            p += 1 if p == 2 else 2
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # 3. Search Strategy
    # The "maximum independent set on complement graph" inspiration suggests looking at the structure of non-coprime pairs.
    # However, for squarefree numbers, the divisor lattice is a Boolean lattice.
    # The number of divisors is 2^k.
    # The score is the number of edges in the sorted list of divisors that connect coprime sets.
    # 
    # Key Insight from "sparse non-coprime" strategy:
    # We want to maximize the number of transitions where gcd is 1.
    # For squarefree n, divisors are subsets of primes.
    # Sorted order is by value.
    # Two adjacent divisors d_i, d_{i+1} are coprime if they share no prime factors.
    # 
    # The current best is likely the primorial of the first 15 primes (or a close variant).
    # Let's start with the primorial of the first 15 primes.
    
    # Product of first 15 primes:
    # 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    # Let's compute it and check.
    
    current_primes = primes[:15]
    n_current = 1
    for p in current_primes:
        n_current *= p
        
    if n_current > 10**18:
        # Reduce primes until it fits
        while n_current > 10**18 and current_primes:
            n_current //= current_primes[-1]
            current_primes.pop()
            
    best_n = n_current
    best_score = calc_score(best_n)
    
    # Local Search: Hill Climbing
    # Moves:
    # 1. Swap a prime p in current_primes with a prime q not in current_primes.
    #    Constraint: new_n <= 10^18.
    # 2. Add a prime q (if n * q <= 10^18).
    # 3. Remove a prime p (if len(current_primes) > 1).
    
    # To make this efficient, we precompute valid candidates for swaps.
    # But since the set of primes changes, we just iterate.
    
    # We will perform a loop of improvement steps.
    # To avoid getting stuck in local minima, we might do some random restarts or larger jumps,
    # but with 140s, simple hill climbing with multiple restarts or a more thorough search might be better.
    
    # Let's implement a simple hill climber that tries all valid single-prime swaps.
    
    improved = True
    while improved:
        improved = False
        # Try swapping each prime
        for i in range(len(current_primes)):
            p_i = current_primes[i]
            # Max q such that n_current * q / p_i <= 10^18
            max_q = int((10**18 * p_i) // n_current)
            
            # Find candidates in primes list
            # We can optimize this by only looking at primes in (p_i, max_q]
            # Since primes is sorted, we can find the range.
            # For simplicity, iterate through primes, but break early if p > max_q.
            # We should skip p_i and any other prime in current_primes.
            
            existing_set = set(current_primes)
            
            for q in primes:
                if q <= p_i:
                    continue
                if q > max_q:
                    break
                if q in existing_set:
                    continue
                
                # Check time
                if time.time() > deadline - 10:
                    return best_n
                
                new_n = n_current // p_i * q
                s = calc_score(new_n)
                
                if s > best_score:
                    best_score = s
                    best_n = new_n
                    current_primes[i] = q
                    n_current = new_n
                    improved = True
                    break # Restart the loop to re-evaluate with new state
            
            if improved:
                break
        
        # Try adding a prime
        if not improved:
            # If we can add a prime without exceeding limit
            # Find smallest prime not in current_primes such that n * p <= 10^18
            existing_set = set(current_primes)
            for q in primes:
                if q in existing_set:
                    continue
                if n_current * q <= 10**18:
                    new_n = n_current * q
                    s = calc_score(new_n)
                    if s > best_score:
                        best_score = s
                        best_n = new_n
                        current_primes.append(q)
                        n_current = new_n
                        improved = True
                    break # Only try the smallest valid one? 
                    # Actually, adding a larger prime might be better?
                    # But usually smaller primes keep the number smaller, allowing more primes later?
                    # Or does adding a larger prime increase score more?
                    # The score depends on the structure.
                    # Let's try the smallest valid one first. If it improves, we do it.
                    # If not, we might want to try others, but that's expensive.
                    # Given the structure, adding a prime doubles the divisor count.
                    # It usually increases the score.
                    # So if the smallest valid one doesn't improve, others might not either?
                    # Not necessarily. But let's stick to greedy for now.
                    # Actually, if adding the smallest doesn't improve, we should probably stop adding.
        
        # Try removing a prime?
        # Removing a prime halves the divisor count. Unlikely to improve score unless it allows adding two larger primes later?
        # But that's a 2-step move.
        # We can ignore removals for now, as the primorial of 15 primes is likely near optimal for count.
        # The product of first 15 primes is ~6.1e17.
        # We have room for another prime up to 1.6e18/6.1e17 ~ 2.6.
        # So we can't add 2 or 3 (already there).
        # So we are stuck at 15 primes unless we swap.
        
        # If no improvement, we might try a random restart or a different heuristic.
        # But let's see if hill climbing converges.
        
        # To prevent infinite loops or slow convergence, we can add a limit to iterations.
        # But the `improved` flag handles termination.
        
    return best_n
