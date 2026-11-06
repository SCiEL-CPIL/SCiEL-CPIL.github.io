"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _get_primes(limit):
    """Generate primes up to limit using sieve."""
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, flag in enumerate(is_prime) if flag]

def _score_fast(primes_list):
    """
    Calculate the score for a square-free number defined by primes_list.
    Returns 0 if product > 10^18.
    """
    n = 1
    for p in primes_list:
        n *= p
        if n > 10**18:
            return -1 # Invalid
    
    k = len(primes_list)
    # Generate all divisors and their bitmasks
    # divisor i corresponds to mask i (binary representation of which primes are included)
    # We need to sort divisors by value.
    divs = []
    for mask in range(1 << k):
        val = 1
        for i in range(k):
            if mask & (1 << i):
                val *= primes_list[i]
        divs.append((val, mask))
        
    divs.sort(key=lambda x: x[0])
    
    count = 0
    for i in range(len(divs) - 1):
        mask1 = divs[i][1]
        mask2 = divs[i+1][1]
        if (mask1 & mask2) == 0:
            count += 1
    return count

def construct():
    # Generate a list of primes up to a reasonable limit
    # P15 is around 6e17. Primes up to 100 are enough for swaps.
    primes = _get_primes(200)
    
    # Initial candidate: first 15 primes
    current_primes = primes[:15]
    current_score = _score_fast(current_primes)
    best_primes = current_primes[:]
    best_score = current_score
    
    # Local search
    improved = True
    iterations = 0
    while improved and iterations < 100:
        improved = False
        iterations += 1
        
        # Try swapping each prime with a larger one
        for i in range(len(current_primes)):
            current_n = 1
            for p in current_primes:
                current_n *= p
            
            # Try replacing current_primes[i] with p_new
            # p_new must be > current_primes[i] and not in current_primes
            # and current_n / current_primes[i] * p_new <= 10^18
            base_n = current_n // current_primes[i]
            limit_p = 10**18 // base_n
            
            for p_new in primes:
                if p_new <= current_primes[i] or p_new > limit_p:
                    continue
                if p_new in current_primes:
                    continue
                
                new_primes = current_primes[:]
                new_primes[i] = p_new
                # Keep sorted? Order doesn't matter for the set, but for mask generation
                # we just need the list.
                
                s = _score_fast(new_primes)
                if s > best_score:
                    best_score = s
                    best_primes = new_primes
                    current_primes = new_primes[:]
                    current_n = 1
                    for p in current_primes:
                        current_n *= p
                    improved = True
                    break # Restart search with new best
                
                # Optional: accept if s > current_score to climb hill
                # But we want global best, so stick to best_score for now
        
        # Try removing 1 prime and adding 2 new primes
        # This increases prime count from 15 to 16?
        # Only possible if we remove a large prime and the product of 2 new small primes fits.
        # Or remove small, add 2 large? No, product increases.
        
        # Actually, if we remove a prime, we have 14 primes.
        # We can add up to 2 new primes if they fit.
        # This would result in 16 primes.
        # Max product of 16 smallest primes is P16 > 10^18.
        # So we can only have 16 primes if we use very small ones?
        # No, P16 is the MINIMUM product for 16 distinct primes.
        # So we can NEVER have 16 distinct primes <= 10^18.
        # So we are stuck with 15 primes.
        
    # Calculate final n
    n = 1
    for p in best_primes:
        n *= p
        
    return n
# EVOLVE-BLOCK-END
