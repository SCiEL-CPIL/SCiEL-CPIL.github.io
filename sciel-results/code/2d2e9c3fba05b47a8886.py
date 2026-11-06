import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    LIMIT = 10**18

    # Precompute primes up to a reasonable bound
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    # We need primes up to maybe 200 or so for flexibility
    all_primes = sieve(500)
    
    def calc_tau_perp(n):
        """Compute tau_perp(n) by generating all divisors, sorting, counting adjacent coprime pairs."""
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = {}
        for p in all_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        
        # Generate all divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
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

    def n_from_primes_with_exps(prime_exp_list):
        """Given list of (prime, exponent), compute n."""
        n = 1
        for p, e in prime_exp_list:
            n *= p ** e
        return n

    # Strategy: 
    # Key insight: For a squarefree n with k distinct primes, the divisor count is 2^k.
    # The structure of coprime adjacent pairs depends on the multiplicative structure.
    # 
    # A better approach: think of n = p1 * p2 * ... * pk (squarefree).
    # Divisors are all subset products. Two adjacent divisors d_i, d_{i+1} in sorted order
    # are coprime iff they share no prime factor.
    #
    # For squarefree n, the divisors sorted: 1, p1, p2, ..., then products of two, etc.
    # The number of coprime adjacent pairs is related to how the subset lattice is ordered.
    #
    # Alternative structural approach: 
    # Use primes with large gaps between them so that divisor products create "gaps"
    # where adjacent divisors tend to be coprime.
    #
    # Actually, let's think differently. The maximum tau_perp likely comes from:
    # - Many distinct prime factors (maximizes number of divisors)
    # - Small exponents (keeps n small, allows more distinct primes)
    # - The specific choice of primes matters for the adjacency structure
    #
    # Let's try: start with squarefree products of the smallest primes, 
    # then try replacing large primes with slightly larger ones to see if score improves.
    
    best_score = -1
    best_n = 2
    
    # First: try squarefree products of first k primes for k = 2 to 15
    for k in range(2, 16):
        n = 1
        for i in range(k):
            n *= all_primes[i]
        if n > LIMIT:
            break
        score = calc_tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n

    # Now try: for the best k found, try different subsets of primes
    # The idea: maybe using non-consecutive primes gives better adjacency structure
    
    # Determine max k for squarefree
    max_k_sq = 0
    prod = 1
    for i, p in enumerate(all_primes):
        if prod * p <= LIMIT:
            prod *= p
            max_k_sq = i + 1
        else:
            break
    
    # Try random subsets of size max_k_sq and max_k_sq - 1, max_k_sq - 2
    random.seed(42)
    tried = set()
    
    for target_k in [max_k_sq, max_k_sq - 1, max_k_sq - 2]:
        if target_k < 2:
            continue
        # We can pick from first ~30 primes to have variety
        pool_size = min(30, len(all_primes))
        pool = all_primes[:pool_size]
        
        num_tries = 200
        for _ in range(num_tries):
            if time.time() > DEADLINE:
                break
            subset = random.sample(pool, target_k)
            subset.sort()
            key = tuple(subset)
            if key in tried:
                continue
            tried.add(key)
            
            n = 1
            for p in subset:
                n *= p
            if n > LIMIT:
                continue
            
            score = calc_tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n

    # Now try: allow small exponents on some primes
    # Start from best squarefree, try bumping exponent of one prime by 1
    # and dropping another prime to stay under LIMIT
    
    # Also try: n = p1^2 * p2 * p3 * ... (one prime squared)
    for k in range(2, 14):
        # Try: 2^2 * 3 * 5 * 7 * ... up to k-1 additional primes
        for start_prime_idx in range(0, 5):  # Which prime gets exponent 2
            n = 1
            used = set()
            # Prime with exponent 2
            p_sq = all_primes[start_prime_idx]
            n *= p_sq * p_sq
            used.add(start_prime_idx)
            
            # Add other primes
            count = 0
            for i in range(len(all_primes)):
                if i in used:
                    continue
                if n * all_primes[i] > LIMIT:
                    break
                n *= all_primes[i]
                used.add(i)
                count += 1
                if count == k - 1:
                    break
            
            if n < 2 or n > LIMIT:
                continue
            score = calc_tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Local search: perturb the best n by swapping primes
    if time.time() < DEADLINE - 5:
        # Factorize best_n
        def factorize(n):
            temp = n
            factors = {}
            for p in all_primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    cnt = 0
                    while temp % p == 0:
                        temp //= p
                        cnt += 1
                    factors[p] = cnt
            if temp > 1:
                factors[temp] = 1
            return factors
        
        best_factors = factorize(best_n)
        
        # Try swapping one prime for another nearby prime
        for _ in range(500):
            if time.time() > DEADLINE:
                break
            
            # Pick a random prime factor to replace
            if not best_factors:
                break
            primes_in_n = list(best_factors.keys())
            p_old = random.choice(primes_in_n)
            e_old = best_factors[p_old]
            
            # Find a replacement prime (not already in n)
            p_new = None
            for candidate in all_primes:
                if candidate not in best_factors and candidate != p_old:
                    # Check if we can swap
                    n_test = best_n // (p_old ** e_old) * (candidate ** e_old)
                    if n_test <= LIMIT and n_test >= 2:
                        p_new = candidate
                        break
            
            if p_new is None:
                # Try a prime not in the initial list
                for candidate in all_primes:
                    if candidate not in best_factors:
                        n_test = best_n // (p_old ** e_old) * (candidate ** e_old)
                        if n_test <= LIMIT and n_test >= 2:
                            p_new = candidate
                            break
            
            if p_new is None:
                continue
            
            n_test = best_n // (p_old ** e_old) * (p_new ** e_old)
            if n_test < 2 or n_test > LIMIT:
                continue
            
            score_test = calc_tau_perp(n_test)
            if score_test > best_score:
                best_score = score_test
                best_n = n_test
                best_factors = factorize(best_n)
    
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
