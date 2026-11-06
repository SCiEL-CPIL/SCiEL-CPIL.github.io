import math
import time

def construct():
    S = 123457
    start_time = time.time()
    
    # We need P(n) for all n in [1, S].
    # P(n) is the largest prime factor of n.
    # We can compute this using a sieve.
    # Initialize an array lp (largest prime) with 0.
    # For each prime p, update lp for its multiples.
    
    limit = S
    # lp[i] will store the largest prime factor of i
    lp = [0] * (limit + 1)
    
    # Sieve of Eratosthenes variant to find largest prime factors
    # We iterate i from 2 to limit.
    # If lp[i] is 0, i is prime.
    # Then for all multiples j of i, if lp[j] is 0, set lp[j] = i.
    # Wait, this sets the *smallest* prime factor if we just set when 0.
    # To get the *largest* prime factor, we can iterate primes in increasing order
    # and overwrite lp[j] with the current prime p. Since we go in increasing order,
    # the last prime that divides j will be the largest.
    
    # Let's refine:
    # Initialize lp to 0.
    # For i from 2 to limit:
    #   if lp[i] == 0: # i is prime
    #     for j in range(i, limit + 1, i):
    #       lp[j] = i
    #       # Since we iterate i in increasing order, the last i that is a factor of j
    #       # is the largest prime factor of j.
    
    for i in range(2, limit + 1):
        if lp[i] == 0: # i is prime
            for j in range(i, limit + 1, i):
                lp[j] = i
                
    # lp[1] is 0, but P(1)=1.
    lp[1] = 1
    
    # Now lp[n] holds the largest prime factor of n for all n in [1, S].
    # We want to minimize max(lp[a], lp[S-a]) for a in [1, S-1].
    
    best_val = float('inf')
    best_pair = [1, S-1]
    
    # Iterate a from 1 to S//2 (since max(P(a), P(S-a)) is symmetric)
    # Actually, we can just iterate all a from 1 to S-1.
    # S is 123457, so 123456 iterations is trivial.
    
    for a in range(1, S):
        b = S - a
        val_a = lp[a]
        val_b = lp[b]
        max_val = val_a if val_a > val_b else val_b
        
        if max_val < best_val:
            best_val = max_val
            best_pair = [a, b]
            
            # Early exit if best_val is 1? No, P(n)=1 only for n=1.
            # If we find a pair where both are 1? Impossible since a+b=S>2.
            # If we find a pair where max is 2? That means both are powers of 2.
            # Sum of two powers of 2 is S.
            # If S is odd, one must be 1 (2^0) and the other S-1.
            # If S-1 is a power of 2, then max(P(1), P(S-1)) = max(1, 2) = 2.
            # Can we get lower? No, P(n) >= 2 for n>1.
            # So if best_val == 2, it's the minimum possible for S>2.
            if best_val == 2:
                break

    # Check time limit (should be well within)
    if time.time() - start_time > 50:
        # Fallback, though unlikely
        return [1, S-1]
        
    return best_pair
