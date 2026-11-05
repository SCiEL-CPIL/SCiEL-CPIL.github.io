"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import gcd

def _sieve(n):
    """Generate primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

def _divisors_from_factors(primes, exponents):
    """Generate all divisors of a number given its prime factorization."""
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            for i in range(e + 1):
                new_divs.append(d * (p ** i))
        divisors = new_divs
    return sorted(divisors)

def _compute_score(divisors):
    """Count adjacent coprime pairs in sorted divisor list."""
    count = 0
    for i in range(len(divisors) - 1):
        if gcd(divisors[i], divisors[i + 1]) == 1:
            count += 1
    return count

def _try_candidate(primes, exponents):
    """Try a candidate number, return (score, n) or (0, 0) if invalid."""
    n = 1
    for p, e in zip(primes, exponents):
        n *= p ** e
    if n < 2 or n > 10**18:
        return (0, 0)
    divs = _divisors_from_factors(primes, exponents)
    score = _compute_score(divs)
    return (score, n)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    # All primes up to 200 for potential combinations
    all_primes = _sieve(200)
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # Fallback
    
    # 1. Start with the known good non-squarefree candidate from current program
    # 2^5 * 3^2 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    primes_base = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    exponents_base = [5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    score, n = _try_candidate(primes_base, exponents_base)
    if score > best_score:
        best_score, best_n = score, n
    
    if time.time() > deadline:
        return best_n

    # 2. Local Search (Inspiration 2)
    # We will maintain the current best exponents and primes.
    # If we start from a non-squarefree candidate, we might want to add primes or adjust exponents.
    # However, adding a prime p to a squarefree number usually increases score significantly.
    # Let's try to expand the set of primes first if possible, or adjust exponents.
    
    # Let's start the local search from the best candidate found so far.
    # We need to be careful: if we add a prime, we change the length of the primes list.
    # If we adjust exponents, we keep the list length.
    
    current_primes = primes_base[:]
    current_exponents = exponents_base[:]
    current_n = best_n
    current_score = best_score
    
    # Helper to update current state
    def update_state(p, e, n, s):
        nonlocal current_primes, current_exponents, current_n, current_score
        nonlocal best_score, best_n
        current_primes = p
        current_exponents = e
        current_n = n
        current_score = s
        if s > best_score:
            best_score = s
            best_n = n

    # Local search loop
    # We will try to:
    # 1. Increase exponents of existing primes (if n allows)
    # 2. Add new primes (if n allows)
    # 3. Decrease exponents to make room for new primes (if beneficial)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try increasing exponents
        for i in range(len(current_primes)):
            if time.time() > deadline:
                return best_n
            p = current_primes[i]
            e = current_exponents[i]
            # Try increasing e by 1
            new_e = e + 1
            new_n = current_n * p
            if new_n > 10**18:
                continue
            
            new_exponents = current_exponents[:]
            new_exponents[i] = new_e
            score, n = _try_candidate(current_primes, new_exponents)
            if score > current_score:
                update_state(current_primes, new_exponents, n, score)
                improved = True
                break # Restart loop with new state
        
        if improved:
            continue
            
        # Try adding a new prime
        # Find the smallest prime not in current_primes
        # We can try primes from all_primes that are not in current_primes
        # To keep it efficient, we can just check primes in increasing order
        # that are not already in the set.
        
        # Optimization: Only try to add primes if we are "squarefree-ish" or if adding a prime
        # generally helps. Adding a prime p to n creates new divisors.
        # Let's try adding the smallest available prime.
        
        # Sort current primes to find gaps? No, just check all_primes.
        added = False
        for p in all_primes:
            if p in current_primes:
                continue
            new_n = current_n * p
            if new_n > 10**18:
                # If we can't add this prime, we might be able to add a larger one? No, larger is worse for n.
                # But maybe we can replace a small prime with a larger one? That's a different move.
                # For now, if we can't add the smallest available prime, we likely can't add any.
                # Wait, current_primes might not be sorted. Let's sort them for consistency in search.
                # But _try_candidate doesn't care about order.
                # However, for "adding next prime", we want the smallest one not present.
                # Let's break if the smallest available prime doesn't fit, assuming larger ones won't.
                # This is a heuristic.
                break
            
            new_primes = current_primes + [p]
            new_exponents = current_exponents + [1]
            score, n = _try_candidate(new_primes, new_exponents)
            if score > current_score:
                update_state(new_primes, new_exponents, n, score)
                improved = True
                added = True
                break
        
        if improved:
            continue

        # Try decreasing exponents to make room for a new prime
        # This is useful if we have high exponents on small primes preventing us from adding more distinct primes.
        # e.g. 2^10 ... vs 2^1 * 3 * ... * 47
        # Usually, more distinct primes is better for tau_perp.
        
        # Try reducing the exponent of the largest prime with exponent > 1?
        # Or the smallest? Reducing the smallest prime (2) frees up the most multiplicative space.
        
        # Let's try reducing exponents of primes in descending order of their contribution to n (log p * e).
        # Actually, just try reducing each exponent > 1 by 1, and see if we can then add a prime.
        
        for i in range(len(current_primes)):
            if time.time() > deadline:
                return best_n
            if current_exponents[i] <= 1:
                continue
            
            p = current_primes[i]
            e = current_exponents[i]
            new_e = e - 1
            new_n = current_n // p
            
            # Now try to add a prime
            # Find smallest prime not in current_primes that fits in new_n
            best_add_score = -1
            best_add_p = -1
            best_add_exponents = None
            best_add_primes = None
            best_add_n = -1
            
            for p_new in all_primes:
                if p_new in current_primes:
                    continue
                if new_n * p_new > 10**18:
                    break
                
                new_primes = current_primes + [p_new]
                new_exponents = current_exponents[:]
                new_exponents[i] = new_e
                new_exponents.append(1)
                
                score, n = _try_candidate(new_primes, new_exponents)
                if score > best_add_score:
                    best_add_score = score
                    best_add_p = p_new
                    best_add_exponents = new_exponents
                    best_add_primes = new_primes
                    best_add_n = n
            
            if best_add_score > current_score:
                update_state(best_add_primes, best_add_exponents, best_add_n, best_add_score)
                improved = True
                break
        
        # If we didn't improve by adding/replacing, try just reducing an exponent?
        # Unlikely to help score directly, but might be a stepping stone.
        # We already tried increasing.
        
        # Another move: Swap a prime?
        # Replace a large prime with a smaller one not in the set?
        # Or replace a small prime with a larger one?
        # Given the time limit, let's stick to Add/Increase/Reduce+Add.
        
    return best_n

# EVOLVE-BLOCK-END
