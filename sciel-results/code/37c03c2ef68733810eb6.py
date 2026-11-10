import time
import math

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 140.0  # Safety margin

    # List of first ~40 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
    
    def tau_perp_score(prime_list):
        """
        Calculate tau_perp for a square-free number given its list of prime factors.
        """
        k = len(prime_list)
        # Generate divisors using bitmasks
        divs = []
        for mask in range(1 << k):
            val = 1
            for i in range(k):
                if mask & (1 << i):
                    val *= prime_list[i]
            divs.append(val)
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for curr in divs[1:]:
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # The maximum number of distinct prime factors for n <= 10^18 is 15.
    # Product of first 15 primes: 2*3*...*47 = 614889782588491410 ~ 6.15 * 10^17
    # Product of first 16 primes: * 53 = 3.26 * 10^19 > 10^18
    # So we are looking for sets of 15 distinct primes.
    
    # Strategy:
    # 1. The "base" set is the first 15 primes.
    # 2. We can try to swap one of the larger primes in the first 15 with a larger prime 
    #    to see if the structure of divisors yields a higher score.
    
    # Base set: first 14 primes
    base_primes = primes[:14]
    base_product = 1
    for p in base_primes:
        base_product *= p
        
    # Candidate 15th primes
    # Must be > 43 to be distinct from base_primes? No, base_primes are 2..43.
    # So 15th prime must be > 43.
    # Also base_product * p <= LIMIT.
    
    best_n = 0
    best_score = -1
    
    # First, evaluate the standard set: first 15 primes
    std_set = primes[:15]
    n_std = 1
    for p in std_set:
        n_std *= p
    if n_std <= LIMIT:
        score_std = tau_perp_score(std_set)
        best_score = score_std
        best_n = n_std
    
    # Now, try variations.
    # We can swap any one of the first 14 primes with a larger prime?
    # Or swap the 15th prime with another larger prime?
    # Or swap two primes?
    
    # Let's iterate through possible 15th primes first (fixing first 14).
    for i in range(14, len(primes)):
        p15 = primes[i]
        if base_product * p15 > LIMIT:
            break
        n = base_product * p15
        score = tau_perp_score(base_primes + [p15])
        if score > best_score:
            best_score = score
            best_n = n
        if time.time() > DEADLINE:
            break
    
    # Now, let's try swapping one of the smaller primes in the base set with a larger one,
    # while keeping 15 distinct primes total.
    # This is a larger search space.
    # We can fix 13 primes and vary 2?
    # Or just do a local search around the best found so far.
    
    # Let's try a heuristic local search:
    # Take the best set found. Try swapping each prime in the set with a larger prime 
    # (not already in the set) such that the product remains <= LIMIT.
    
    if best_n > 0:
        # Factor best_n to get the set of primes (it's square-free)
        temp_n = best_n
        current_set = []
        d = 2
        while d * d <= temp_n:
            if temp_n % d == 0:
                current_set.append(d)
                while temp_n % d == 0:
                    temp_n //= d
            d += 1
        if temp_n > 1:
            current_set.append(temp_n)
            
        # Ensure sorted
        current_set.sort()
        
        improved = True
        while improved and time.time() < DEADLINE:
            improved = False
            current_score = tau_perp_score(current_set)
            
            # Try swapping each prime in current_set with a larger prime
            for idx, p_old in enumerate(current_set):
                # Find candidate replacement primes > p_old
                # We need to check if p_new is in current_set. If so, skip.
                # Also product / p_old * p_new <= LIMIT.
                limit_for_p_new = LIMIT * p_old // best_n # This is approximate, better to compute directly
                
                # Iterate through primes larger than p_old
                # We can limit the search to primes not too large to keep it fast
                for p_new in primes:
                    if p_new <= p_old:
                        continue
                    if p_new in current_set:
                        continue
                    
                    new_product = best_n // p_old * p_new
                    if new_product > LIMIT:
                        break # Since primes are sorted, larger p_new will also exceed limit
                    
                    new_set = current_set.copy()
                    new_set[idx] = p_new
                    new_set.sort()
                    
                    score_new = tau_perp_score(new_set)
                    if score_new > best_score:
                        best_score = score_new
                        best_n = new_product
                        current_set = new_set
                        improved = True
                        # Restart the while loop with the new best set
                        break
                if improved:
                    break
            
            # Additional heuristic: try adding a prime and removing a smaller one? 
            # This is covered by the swap logic above if we consider "adding" as swapping a large prime for a larger one?
            # No, swap logic keeps size constant.
            # We established max size is 15.
            # So we are optimizing the choice of 15 primes.
            
            # To be more thorough, we can try swapping TWO primes simultaneously?
            # That's O(k^2 * P^2) which might be too slow.
            # Let's stick to single swaps for now, but maybe do a few rounds.
            
            # Let's also try swapping a small prime with a large prime to "stretch" the divisors?
            # The swap logic above does this.
            
            # If no improvement in a full pass, we might be at a local maximum.
            # Given the small search space (15 primes from ~30 candidates), 
            # and the fact that we started with the smallest primes, 
            # the local search should find the global optimum or very close.
            
            # Let's add a check for time.
            if time.time() > DEADLINE - 10:
                break

    if best_n == 0:
        # Fallback to first 15 primes
        best_n = 1
        for p in primes[:15]:
            best_n *= p
            
    assert 2 <= best_n <= LIMIT
    return best_n
