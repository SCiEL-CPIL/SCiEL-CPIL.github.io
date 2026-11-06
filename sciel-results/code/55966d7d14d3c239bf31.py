import math
import time
import heapq
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    LIMIT = 10**18

    # Sieve for primes up to a reasonable limit
    # We need enough primes to potentially fill the budget.
    # Product of first 15 primes is ~ 6e17. 
    # If we use larger primes, we can fit fewer.
    # Let's sieve up to 100,000 to have plenty of options.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    all_primes = sieve(10000)
    
    # We only care about primes that can fit in the budget.
    # Actually, any prime < LIMIT is valid, but if we have k primes, their product must be <= LIMIT.
    # Small primes are "cheap" in log space, allowing more factors.
    # Large primes are "expensive".
    
    # To calculate tau_perp efficiently during the greedy process,
    # we need to maintain the sorted list of divisors.
    # However, maintaining a sorted list and checking all adjacencies is O(N log N) where N is num divisors.
    # N can be up to 2^15 = 32768. This is small enough.
    
    # Strategy:
    # 1. Start with n=1. Divisors = [1]. Score = 0.
    # 2. Candidate primes: all primes such that n * p <= LIMIT.
    # 3. For each candidate prime p, calculate the "gain" in tau_perp if we multiply n by p.
    #    Since we want to maximize the final score, and the cost is multiplicative (log-additive),
    #    a simple greedy "max gain" might fail. 
    #    "Gain per log unit" is a better heuristic for knapsack-like problems.
    #    Gain = new_score - old_score.
    #    Cost = log(p).
    #    Ratio = Gain / Cost.
    #    Pick the prime with highest Ratio.
    # 4. Update n, divisors, score.
    # 5. Repeat until no more primes can be added or time runs out.
    
    # Optimization: 
    # Calculating the exact gain for every prime at every step is expensive.
    # How many primes to consider? 
    # If n is small, many primes fit. 
    # If n is large, few primes fit.
    # Let's limit the candidate pool to the first 200 primes and any primes that are "just" under LIMIT/n.
    # Actually, small primes are generally better for maximizing count of factors, which tends to increase total divisors and potential coprime pairs.
    # Let's restrict the search space to primes < 1000 to keep the candidate set manageable, 
    # unless n is very small, in which case we might consider larger primes if they offer a huge structural jump?
    # Unlikely. The structure of divisors is dominated by the small primes.
    
    # Let's refine the "Gain" calculation.
    # If current divisors are D, and we multiply by prime p (not in D), 
    # the new divisors are D union {d*p for d in D}.
    # The new sorted list is the merge of D and D*p.
    # We need to count adjacent pairs (a, b) in the merged list such that gcd(a,b)=1.
    
    def get_divisors_from_factors(factors):
        """
        factors: list of (prime, exponent)
        Returns sorted list of divisors.
        """
        divs = [1]
        for p, e in factors:
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        divs.sort()
        return divs

    def calc_tau_perp(divs):
        """
        Calculate tau_perp for a sorted list of divisors.
        """
        if len(divs) < 2:
            return 0
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # State:
    # current_n
    # current_factors: list of (prime, exponent)
    # current_divs: sorted list of divisors
    # current_score: tau_perp
    
    current_n = 1
    current_factors = []
    current_divs = [1]
    current_score = 0
    
    # Candidate pool:
    # We will iterate through primes in increasing order.
    # At each step, we check which primes can be added (n * p <= LIMIT).
    # We evaluate the "value" of adding p.
    
    # To speed up, we can precompute primes up to 1000.
    # If n * p > LIMIT for p > 1000, we stop.
    # If n is small, we might want to check primes > 1000?
    # Probably not beneficial for maximizing *count* of coprime pairs, as small primes create finer structure.
    # Let's stick to primes < 1000 for the greedy selection, but allow "swapping" later?
    # No, let's keep it simple first: Greedy addition from a pool of small primes.
    
    pool_primes = [p for p in all_primes if p < 1000]
    
    # Greedy Loop
    while time.time() < DEADLINE:
        best_p = -1
        best_ratio = -1
        best_new_score = -1
        best_new_divs = None
        best_new_n = -1
        best_new_factors = None
        
        # Check all candidate primes
        for p in pool_primes:
            if current_n * p > LIMIT:
                continue
            
            # Simulate adding p
            # new_factors = current_factors + [(p, 1)]
            # But if p is already in factors, exponent increases.
            # Since we start with 1 and only add primes not yet in the "base" set?
            # Wait, the greedy approach might add p, then later add p again?
            # If we add p, it's in the factors. If we add it again, exponent becomes 2.
            # Let's allow this.
            
            new_factors = list(current_factors)
            found = False
            for i, (prime, exp) in enumerate(new_factors):
                if prime == p:
                    new_factors[i] = (prime, exp + 1)
                    found = True
                    break
            if not found:
                new_factors.append((p, 1))
            
            new_divs = get_divisors_from_factors(new_factors)
            new_score = calc_tau_perp(new_divs)
            new_n = current_n * p
            
            # Heuristic: Score / log(n) or just Score?
            # The task is to maximize the final score.
            # If we pick a prime that gives a small immediate gain but uses little budget,
            # we might fit more primes later.
            # Ratio = new_score / math.log(new_n)
            # Or marginal gain: (new_score - current_score) / math.log(p)
            
            marginal_gain = new_score - current_score
            if marginal_gain <= 0:
                # Adding this prime doesn't help?
                # It might still be useful if it enables future moves? Unlikely in this metric.
                # Actually, adding a prime always increases the number of divisors.
                # Does it always increase tau_perp?
                # Not necessarily. It might cluster divisors.
                # If marginal_gain <= 0, skip?
                # Let's allow small negative gains if it's the only option? No, better to stop or pick something else.
                # For now, skip non-positive marginal gains.
                continue
            
            cost = math.log(p)
            ratio = marginal_gain / cost
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_p = p
                best_new_score = new_score
                best_new_divs = new_divs
                best_new_n = new_n
                best_new_factors = new_factors
        
        if best_p == -1:
            # No prime improves the score via this heuristic
            break
            
        # Commit the best move
        current_n = best_new_n
        current_factors = best_new_factors
        current_divs = best_new_divs
        current_score = best_new_score
        
        # Check if we can't add any more primes from the pool
        # If the smallest prime in pool doesn't fit, we are done with this pool.
        # Should we expand the pool?
        # If current_n is small, we might have skipped large primes.
        # But large primes are "expensive".
        # Let's assume the pool of < 1000 is sufficient for the "dense" structure.
        
        # If we've added all primes in the pool, or the smallest one doesn't fit, break.
        # Actually, the loop naturally breaks if no prime fits (best_p remains -1 if all n*p > LIMIT).
        # But what if some primes fit but don't improve score?
        # Then we break. This might be suboptimal if a "bad" move enables a "good" one?
        # Unlikely for this specific metric.
        
        # Let's also try a different heuristic: Maximize absolute score gain?
        # The ratio heuristic is for knapsack. 
        # Let's run a second pass with "Max Absolute Gain" if we have time?
        # Or just stick to one robust heuristic.
        
        # Refinement: What if we get stuck?
        # Try "Max Absolute Gain" instead of Ratio.
        # Let's alternate or pick based on remaining budget?
        # For now, Ratio is standard for this type of packing.
        
        # If we have exhausted the pool or can't add more, we stop.
        # But wait, what if we want to add a prime > 1000?
        # If current_n is very small (e.g., 2), we could add 9973.
        # Would that be better?
        # 2 * 9973 = 19946. Divisors: 1, 2, 9973, 19946.
        # Pairs: (1,2) gcd1, (2,9973) gcd1, (9973,19946) gcd9973!=1. Score 2.
        # vs 2 * 3 = 6. Divisors: 1, 2, 3, 6.
        # Pairs: (1,2) gcd1, (2,3) gcd1, (3,6) gcd3!=1. Score 2.
        # Same score, but 6 uses less budget. So small primes are better.
        
        # So restricting to small primes is likely optimal.
        
    # After greedy construction, try local optimization:
    # Swap a prime factor with another prime to see if score improves.
    # Or increase an exponent?
    
    # Let's do a simple local search on the final configuration.
    # Represent current state.
    # Try swapping each prime factor p_i with a nearby prime q.
    # Or try increasing exponent of p_i (if budget allows).
    
    # Time check
    if time.time() < DEADLINE - 5:
        # Local Search: Increase exponents
        # For each prime in current_factors, try increasing its exponent by 1.
        # This is equivalent to multiplying by p again.
        # We already did this in the greedy loop (allowing re-addition).
        # So the greedy loop should have handled exponent increases if they were beneficial.
        
        # What about swapping?
        # Replace p_i with q.
        # n_new = n / p_i * q.
        # Check if n_new <= LIMIT.
        # Check if score improves.
        
        # Get list of primes in current_factors
        active_primes = [p for p, e in current_factors]
        
        # Candidate replacements: primes near the active ones, or small primes not in active set.
        # To keep it fast, only check a few swaps.
        
        swapped = False
        for _ in range(100): # Limit swaps
            if time.time() > DEADLINE - 2:
                break
            
            best_swap_n = -1
            best_swap_score = current_score
            best_swap_factors = None
            best_swap_divs = None
            
            # Try swapping each active prime
            for i, (p_old, e_old) in enumerate(current_factors):
                # Try replacing with a prime p_new
                # Range of p_new: small primes
                for p_new in pool_primes[:50]: # Only first 50 primes
                    if p_new == p_old:
                        continue
                    
                    # Remove p_old once, add p_new once
                    # This changes the factorization.
                    new_factors = list(current_factors)
                    # Remove one instance of p_old
                    new_factors[i] = (p_old, e_old - 1)
                    if new_factors[i][1] == 0:
                        new_factors.pop(i)
                    else:
                        # Keep the rest
                        pass
                    
                    # Add p_new
                    found_new = False
                    for j, (prime, exp) in enumerate(new_factors):
                        if prime == p_new:
                            new_factors[j] = (prime, exp + 1)
                            found_new = True
                            break
                    if not found_new:
                        new_factors.append((p_new, 1))
                    
                    n_new = current_n // p_old * p_new
                    if n_new > LIMIT or n_new < 2:
                        continue
                    
                    new_divs = get_divisors_from_factors(new_factors)
                    new_score = calc_tau_perp(new_divs)
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_n = n_new
                        best_swap_factors = new_factors
                        best_swap_divs = new_divd if 'new_divd' in dir() else new_divs # Typo fix below
                        
            if best_swap_n != -1 and best_swap_score > current_score:
                current_n = best_swap_n
                current_factors = best_swap_factors
                current_divs = best_swap_divs
                current_score = best_swap_score
                swapped = True
            else:
                break # No improvement found in this pass

    return current_n

# Fix typo in previous block logic before finalizing
# I will rewrite the local search part cleanly in the final code.

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    LIMIT = 10**18

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    all_primes = sieve(10000)
    pool_primes = [p for p in all_primes if p < 1000]

    def get_divisors_from_factors(factors):
        divs = [1]
        for p, e in factors:
            if e == 0:
                continue
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        divs.sort()
        return divs

    def calc_tau_perp(divs):
        if len(divs) < 2:
            return 0
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    current_n = 1
    current_factors = []
    current_divs = [1]
    current_score = 0

    # Greedy Construction
    while time.time() < DEADLINE - 10:
        best_p = -1
        best_ratio = -1
        best_new_score = -1
        best_new_divs = None
        best_new_n = -1
        best_new_factors = None
        
        for p in pool_primes:
            if current_n * p > LIMIT:
                continue
            
            new_factors = list(current_factors)
            found = False
            for i, (prime, exp) in enumerate(new_factors):
                if prime == p:
                    new_factors[i] = (prime, exp + 1)
                    found = True
                    break
            if not found:
                new_factors.append((p, 1))
            
            new_divs = get_divisors_from_factors(new_factors)
            new_score = calc_tau_perp(new_divs)
            new_n = current_n * p
            
            marginal_gain = new_score - current_score
            if marginal_gain <= 0:
                continue
            
            cost = math.log(p)
            ratio = marginal_gain / cost
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_p = p
                best_new_score = new_score
                best_new_divs = new_divs
                best_new_n = new_n
                best_new_factors = new_factors
        
        if best_p == -1:
            break
            
        current_n = best_new_n
        current_factors = best_new_factors
        current_divs = best_new_divs
        current_score = best_new_score

    # Local Search: Swap primes
    if time.time() < DEADLINE - 5:
        improved = True
        while improved and time.time() < DEADLINE - 2:
            improved = False
            best_swap_n = -1
            best_swap_score = current_score
            best_swap_factors = None
            best_swap_divs = None
            
            for i, (p_old, e_old) in enumerate(current_factors):
                # Try replacing one instance of p_old with p_new
                for p_new in pool_primes[:100]:
                    if p_new == p_old:
                        continue
                    
                    new_factors = list(current_factors)
                    # Decrement p_old
                    new_factors[i] = (p_old, e_old - 1)
                    if new_factors[i][1] == 0:
                        new_factors.pop(i)
                    
                    # Increment p_new
                    found_new = False
                    for j, (prime, exp) in enumerate(new_factors):
                        if prime == p_new:
                            new_factors[j] = (prime, exp + 1)
                            found_new = True
                            break
                    if not found_new:
                        new_factors.append((p_new, 1))
                    
                    n_new = current_n // p_old * p_new
                    if n_new > LIMIT or n_new < 2:
                        continue
                    
                    new_divs = get_divisors_from_factors(new_factors)
                    new_score = calc_tau_perp(new_divs)
                    
                    if new_score > best_swap_score:
                        best_swap_score = new_score
                        best_swap_n = n_new
                        best_swap_factors = new_factors
                        best_swap_divs = new_divs
            
            if best_swap_n != -1 and best_swap_score > current_score:
                current_n = best_swap_n
                current_factors = best_swap_factors
                current_divs = best_swap_divs
                current_score = best_swap_score
                improved = True

    return current_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
