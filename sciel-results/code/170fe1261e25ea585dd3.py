import time
import math
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(2000)
    prime_set = set(primes)
    
    # 2. Fast tau_perp using bitmask representation
    # For square-free numbers, each divisor corresponds to a subset of primes.
    # We can represent each divisor by a bitmask of which primes divide it.
    # gcd(d_i, d_{i+1}) == 1 iff (mask_i & mask_{i+1}) == 0
    
    score_cache = {}
    
    def tau_perp_fast(prime_tuple):
        """
        Calculate tau_perp for a square-free number using bitmask technique.
        """
        if prime_tuple in score_cache:
            return score_cache[prime_tuple]
            
        k = len(prime_tuple)
        if k == 0:
            return 0
            
        # Generate divisors in sorted order using a priority queue or iterative merge
        # For efficiency, generate all 2^k divisors with their bitmasks, then sort by value
        
        # Build divisor list: each entry is (value, bitmask)
        divs = [(1, 0)]
        for idx, p in enumerate(prime_tuple):
            bit = 1 << idx
            new_divs = [(v * p, m | bit) for v, m in divs]
            divs.extend(new_divs)
        
        # Sort by divisor value
        divs.sort(key=lambda x: x[0])
        
        # Count adjacent coprime pairs using bitmask AND
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
                
        score_cache[prime_tuple] = score
        return score

    # 3. Find max number of distinct primes that fit in 10^18
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # 4. Multi-start initialization: try several different initial sets
    candidates = []
    
    # Candidate 1: first max_k primes
    init1 = tuple(primes[:max_k])
    candidates.append(init1)
    
    # Candidate 2: first (max_k-1) primes + a larger prime if it fits
    if max_k >= 2:
        base_prod = 1
        for p in primes[:max_k-1]:
            base_prod *= p
        # Find largest prime that fits
        max_p = 10**18 // base_prod
        # Find largest prime <= max_p not already in base
        for p in reversed(primes):
            if p <= max_p and p not in primes[:max_k-1]:
                init2 = tuple(sorted(primes[:max_k-1] + [p]))
                candidates.append(init2)
                break
    
    # Candidate 3: skip some small primes, use medium primes
    if max_k >= 4:
        # Try primes[1:max_k+1] (skip 2)
        prod_test = 1
        valid = True
        for p in primes[1:max_k+1]:
            prod_test *= p
            if prod_test > 10**18:
                valid = False
                break
        if valid:
            candidates.append(tuple(primes[1:max_k+1]))
    
    # Candidate 4: random valid set
    if max_k >= 3:
        for _ in range(5):
            random_set = set()
            prod_test = 1
            for p in random.sample(primes[:100], min(max_k, 100)):
                if prod_test * p <= 10**18:
                    random_set.add(p)
                    prod_test *= p
            if len(random_set) >= 3:
                candidates.append(tuple(sorted(random_set)))
    
    # Evaluate all candidates and pick the best
    best_primes = None
    best_score = -1
    for cand in candidates:
        s = tau_perp_fast(cand)
        if s > best_score:
            best_score = s
            best_primes = list(cand)
    
    if best_primes is None:
        best_primes = list(primes[:max_k])
        best_score = tau_perp_fast(tuple(best_primes))
    
    current_prod = 1
    for p in best_primes:
        current_prod *= p
    
    # 5. Systematic local search with all-swap evaluation per pass
    improved = True
    passes = 0
    while improved and time.time() < deadline - 15.0:
        improved = False
        passes += 1
        best_primes.sort()
        current_set = set(best_primes)
        
        # For each position, try all valid swaps
        best_swap = None  # (i, p_in, new_score)
        
        for i in range(len(best_primes)):
            if time.time() > deadline - 15.0:
                break
                
            p_out = best_primes[i]
            max_p_in = int(10**18 * p_out / current_prod)
            
            # Try each candidate prime
            for p_in in primes:
                if p_in > max_p_in:
                    break
                if p_in in current_set:
                    continue
                    
                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()
                new_tuple = tuple(new_primes)
                
                s = tau_perp_fast(new_tuple)
                if s > best_score:
                    best_swap = (i, p_in, s)
                    # Don't break - continue to find the best swap in this pass
        
        # Apply the best swap found in this pass
        if best_swap is not None:
            i, p_in, new_score = best_swap
            p_out = best_primes[i]
            best_primes[i] = p_in
            best_primes.sort()
            current_prod = current_prod * p_in // p_out
            best_score = new_score
            improved = True
        else:
            # No improvement from swaps. Try adding or removing a prime.
            
            # Try adding a prime (if we have room)
            # Find smallest prime not in current_set that can be added
            for p_add in primes:
                if p_add in current_set:
                    continue
                if current_prod * p_add > 10**18:
                    break
                new_primes = best_primes + [p_add]
                new_primes.sort()
                s = tau_perp_fast(tuple(new_primes))
                if s > best_score:
                    best_primes = new_primes
                    current_prod *= p_add
                    best_score = s
                    improved = True
                    break
            
            # If no improvement from adding, try removing the smallest prime
            if not improved and len(best_primes) > 2:
                # Try removing each prime and see if a larger one can be added
                for i in range(len(best_primes)):
                    p_remove = best_primes[i]
                    prod_without = current_prod // p_remove
                    # Find largest prime that fits
                    max_p = 10**18 // prod_without
                    for p_add in reversed(primes):
                        if p_add <= max_p and p_add not in current_set:
                            new_primes = best_primes[:]
                            new_primes[i] = p_add
                            new_primes.sort()
                            s = tau_perp_fast(tuple(new_primes))
                            if s > best_score:
                                best_primes = new_primes
                                current_prod = prod_without * p_add
                                best_score = s
                                improved = True
                                break
                    if improved:
                        break
    
    # 6. Final random perturbation if time remains
    if time.time() < deadline - 5.0:
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            
            curr_primes = best_primes[:]
            curr_prod = current_prod
            curr_set = set(curr_primes)
            
            # Do 2-4 random valid swaps
            n_swaps = random.randint(2, 4)
            for _ in range(n_swaps):
                i = random.randint(0, len(curr_primes) - 1)
                p_out = curr_primes[i]
                max_p_in = int(10**18 * p_out / curr_prod)
                
                candidates = [p for p in primes if p <= max_p_in and p not in curr_set]
                if not candidates:
                    continue
                    
                p_in = random.choice(candidates)
                curr_primes[i] = p_in
                curr_prod = curr_prod * p_in // p_out
                curr_set.discard(p_out)
                curr_set.add(p_in)
                curr_primes.sort()
            
            s = tau_perp_fast(tuple(curr_primes))
            if s > best_score:
                best_score = s
                best_primes = curr_primes
                current_prod = curr_prod

    # Construct final number
    result = 1
    for p in best_primes:
        result *= p
        
    return result
