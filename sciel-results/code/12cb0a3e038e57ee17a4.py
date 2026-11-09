"""Construct an integer n <= 10^18 maximizing the number of adjacent coprime divisor pairs."""
# EVOLVE-BLOCK-START

def construct():
    # The goal is to maximize tau_perp(n), the number of adjacent pairs of divisors
    # (in increasing order) that are coprime.
    #
    # Key Insight:
    # The number of divisors tau(n) is the product of (e_i + 1) for the prime factorization
    # n = p1^e1 * p2^e2 * ... * pk^ek.
    # To maximize the structural diversity and the likelihood of coprime adjacent pairs,
    # we want to maximize the number of divisors tau(n) within the limit 10^18.
    # A highly composite number (HCN) or a number with a large number of divisors
    # (superior highly composite) tends to have a dense divisor set.
    #
    # Strategy:
    # We perform a search over the exponents of the smallest primes to find the number
    # n <= 10^18 that maximizes tau(n).
    # It is known that the optimal exponents for maximizing tau(n) under a size constraint
    # are non-increasing: e1 >= e2 >= ... >= ek.
    # Also, the primes used must be the smallest consecutive primes starting from 2.
    
    import math
    
    limit = 10**18
    
    # We need a list of small primes.
    # The product of the first 15 primes is approx 10^17, so 15-20 primes might be involved
    # if exponents are 1. But with higher exponents on small primes, fewer distinct primes are used.
    # Let's generate the first 30 primes to be safe.
    
    def generate_primes(n):
        sieve = [True] * (n*10)
        primes = []
        for i in range(2, n*10):
            if sieve[i]:
                primes.append(i)
                for j in range(i*i, n*10, i):
                    sieve[j] = False
            if len(primes) == n:
                break
        return primes

    primes = generate_primes(30)
    
    best_n = 1
    best_tau = 0
    
    # Recursive search for exponents
    # index: current prime index
    # current_n: current product
    # current_tau: current number of divisors
    # max_exp: maximum allowed exponent for this prime (to enforce non-increasing exponents)
    
    def search(index, current_n, current_tau, max_exp):
        nonlocal best_n, best_tau
        
        # Pruning: if current_tau is already smaller than best_tau, and even if we added
        # many more primes with exponent 1, we might not beat it?
        # Actually, just checking if we can improve is tricky. 
        # However, if current_n * next_prime > limit, we stop.
        
        if index >= len(primes):
            if current_tau > best_tau:
                best_tau = current_tau
                best_n = current_n
            return
        
        p = primes[index]
        
        # Try exponents from max_exp down to 0
        # We must try 0 to allow skipping this prime (though for HCNs we usually don't skip
        # small primes if we use larger ones, but the recursive structure handles this by
        # setting max_exp for the next prime to the current exponent).
        # Actually, if we set exponent 0 for p_i, we must set 0 for all subsequent p_j.
        # So if we pick e_i = 0, we should just update best and return, or rather,
        # we should not recurse to the next prime with max_exp 0? 
        # No, if e_i = 0, then e_{i+1} must be 0. So we can just update best_n/tau here
        # and not recurse further.
        
        for e in range(max_exp, -1, -1):
            if e > 0:
                if current_n * (p ** e) > limit:
                    continue
                new_n = current_n * (p ** e)
                new_tau = current_tau * (e + 1)
                
                # If new_tau is already less than best_tau, we might still want to continue?
                # No, because adding more primes will increase tau. 
                # Wait, if we pick a small exponent here, we allow larger exponents for smaller primes?
                # No, the constraint is e_i >= e_{i+1}.
                # So if we pick e, the next max_exp is e.
                
                # Optimization: If new_tau * (max_exp + 1) ** (len(primes) - index - 1) < best_tau, we could prune?
                # This is an upper bound on how much tau can grow.
                
                search(index + 1, new_n, new_tau, e)
            else:
                # If e == 0, we stop adding primes.
                if current_tau > best_tau:
                    best_tau = current_tau
                    best_n = current_n
                break # No point in trying lower exponents (none) or recursing further with 0 constraint

    # Initial call:
    # Max exponent for 2 is bounded by log2(10^18) ~ 60.
    # Let's start with 60.
    search(0, 1, 1, 60)
    
    return best_n

# EVOLVE-BLOCK-END
