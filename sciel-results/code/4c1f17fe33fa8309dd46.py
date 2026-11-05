import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Precompute primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(500)
    # Filter primes that could reasonably appear in n <= 10^18
    # If n has k primes, the product of the first k primes must be <= 10^18.
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 6.14e17
    # 2*...*53 = 3.25e19 > 10^18. So max 15 distinct primes.
    # However, if we use large primes, we can have fewer.
    # We will consider primes up to 1000 for the search space.
    
    # DP State:
    # We want to compute tau_perp(n) for n = p1^e1 * ... * pk^ek.
    # Divisors are of the form prod p_i^a_i.
    # Two divisors d, d' are coprime iff they share no prime factors.
    # This is equivalent to: the set of primes dividing d and the set of primes dividing d' are disjoint.
    # Let S(d) be the bitmask of primes dividing d.
    # We sort divisors.
    # The DP approach described in the prompt is a bit abstract. 
    # Actually, for k <= 15, generating all divisors is O(tau(n)).
    # Max tau(n) for n <= 10^18 is around 103680.
    # Generating 100k divisors is fast in Python if optimized, but sorting is O(N log N).
    # The "Bitmask DP" idea might be to count coprime pairs without explicit sorting?
    # No, the score depends on *adjacent* pairs in the *sorted* list. 
    # So we MUST know the order. 
    # However, we can compute the score faster if we avoid Python overhead of list appends and gcd calls.
    #
    # Let's try a different optimization: 
    # 1. Precompute divisors and their prime masks.
    # 2. Sort divisors.
    # 3. Count coprime pairs using bit operations (mask_i & mask_j == 0).
    # This is faster than math.gcd.
    
    def calc_score_fast(factors):
        # factors: dict {prime: exponent}
        if not factors:
            return 0 # n=1, but n>=2
        
        # Generate divisors and their masks
        # We need to map prime index to bit position
        # To make it fast, we assume the set of primes is small (<= 15)
        
        # Get list of (prime, exponent)
        prime_list = list(factors.keys())
        k = len(prime_list)
        if k == 0:
            return 0
            
        # Map prime to index
        prime_to_idx = {p: i for i, p in enumerate(prime_list)}
        
        # Generate all divisors
        divs = [1]
        masks = [0]
        
        for i, p in enumerate(prime_list):
            e = factors[p]
            new_divs = []
            new_masks = []
            p_power = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_power)
                    new_masks.append(masks[divs.index(d)] | (1 << i)) # This index lookup is slow
                p_power *= p
            divs = new_divs
            masks = new_masks
            
        # The index lookup in the loop above is O(N). Let's optimize generation.
        
    def calc_score_optimized(factors):
        if not factors:
            return 0
        
        prime_list = list(factors.keys())
        k = len(prime_list)
        
        # Generate divisors and masks efficiently
        divs = [1]
        masks = [0]
        
        for i, p in enumerate(prime_list):
            e = factors[p]
            new_divs = []
            new_masks = []
            p_power = 1
            for _ in range(e + 1):
                for j in range(len(divs)):
                    new_divs.append(divs[j] * p_power)
                    new_masks.append(masks[j] | (1 << i))
                p_power *= p
            divs = new_divs
            masks = new_masks
            
        # Sort by divisor value
        # We need to pair divs and masks.
        # Create list of tuples, sort, then extract masks
        pairs = list(zip(divs, masks))
        pairs.sort(key=lambda x: x[0])
        
        # Count coprime adjacent pairs
        score = 0
        for i in range(len(pairs) - 1):
            if (pairs[i][1] & pairs[i+1][1]) == 0:
                score += 1
                
        return score

    # Initial Candidate: Product of first 15 primes
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    initial_primes = primes[:15]
    current_factors = {p: 1 for p in initial_primes}
    current_n = 1
    for p in initial_primes:
        current_n *= p
        
    current_score = calc_score_optimized(current_factors)
    
    best_n = current_n
    best_score = current_score
    best_factors = dict(current_factors)
    
    # Local Search
    # Moves:
    # 1. Increase exponent of p
    # 2. Decrease exponent of p
    # 3. Add prime q
    # 4. Remove prime p
    
    # Candidate primes for addition
    # We should consider primes up to a certain limit.
    # If n is small, we can add large primes.
    # But adding a large prime only increases divisors by 2 (1 and p).
    # It might break a coprime chain or create a new one.
    # Usually, smaller primes are better for structure.
    
    candidate_primes = primes[:50] # Up to 229
    
    improved = True
    iteration = 0
    max_iterations = 500 # Increased iterations due to faster calc
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        cur_primes = list(best_factors.keys())
        
        best_move_score = best_score
        best_move_n = best_n
        best_move_factors = best_factors
        
        # 1. Increase exponent
        for p in cur_primes:
            if time.time() > deadline:
                break
            new_n = best_n * p
            if new_n <= 10**18:
                new_factors = dict(best_factors)
                new_factors[p] += 1
                s = calc_score_optimized(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = new_factors
                    improved = True
        
        # 2. Decrease exponent
        for p in cur_primes:
            if time.time() > deadline:
                break
            if best_factors[p] > 1:
                new_n = best_n // p
                new_factors = dict(best_factors)
                new_factors[p] -= 1
                s = calc_score_optimized(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = new_factors
                    improved = True
        
        # 3. Add new prime
        for q in candidate_primes:
            if time.time() > deadline:
                break
            if q not in best_factors:
                new_n = best_n * q
                if new_n <= 10**18:
                    new_factors = dict(best_factors)
                    new_factors[q] = 1
                    s = calc_score_optimized(new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_move_n = new_n
                        best_move_factors = new_factors
                        improved = True
        
        # 4. Remove prime
        for p in cur_primes:
            if time.time() > deadline:
                break
            if best_factors[p] == 1:
                new_n = best_n // p
                new_factors = dict(best_factors)
                del new_factors[p]
                s = calc_score_optimized(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = new_factors
                    improved = True
        
        if improved:
            best_n = best_move_n
            best_score = best_move_score
            best_factors = best_move_factors
        else:
            # If no improvement, try a random restart or perturbation?
            # For now, break to avoid infinite loop if stuck in local optimum.
            # But we have max_iterations.
            # Let's try a few random perturbations if stuck.
            if iteration > 10:
                break

    return best_n

# Test locally
if __name__ == "__main__":
    n = construct()
    print(f"Constructed n: {n}")
    
    # Verify
    factors = {}
    temp_n = n
    p = 2
    while p * p <= temp_n:
        if temp_n % p == 0:
            count = 0
            while temp_n % p == 0:
                temp_n //= p
                count += 1
            factors[p] = count
        p += 1 if p == 2 else 2
    if temp_n > 1:
        factors[temp_n] = 1
        
    score = calc_score_optimized(factors)
    print(f"Score: {score}")
