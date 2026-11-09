"""Optimized program: Sieve-based smoothness check, broader search."""
# EVOLVE-BLOCK-START

import math
import time

def construct():
    # ---------- Sieve of Eratosthenes ----------
    # We need primes up to a limit to perform trial division.
    # Let's sieve up to 5_000_000 to be safe, but we will only use primes <= p for checking p.
    SIEVE_LIMIT = 5_000_000
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SIEVE_LIMIT ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    # ---------- Smoothness Check ----------
    def is_p_smooth(val, p, primes):
        """
        Returns True if all prime factors of val are <= p.
        Uses trial division by primes up to min(p, sqrt(val)).
        """
        if val <= p:
            return True
        # We need to divide by all primes q such that q <= min(p, sqrt(val))
        # If after division val > 1, it must be prime (as it has no factors <= sqrt(val))
        # and we check if it is <= p.
        
        # Optimization: if val is prime and > p, return False.
        # We iterate through primes.
        limit = min(p, int(math.isqrt(val)))
        
        # Find index in primes list up to limit
        # Since primes is sorted, we can just iterate and break when prime > limit
        for q in primes:
            if q > limit:
                break
            if val % q == 0:
                while val % q == 0:
                    val //= q
                if val == 1:
                    return True
                # Update limit? No, sqrt(val) decreases.
                # But we can just continue with the same primes list, 
                # breaking when q > new_limit or q > p.
                # Actually, if val becomes 1, we return.
                # If val > 1, we continue.
                # The condition q <= limit was based on original val.
                # If val decreases, sqrt(val) decreases.
                # So we can break earlier.
                new_limit = min(p, int(math.isqrt(val)))
                if q > new_limit:
                    break
                # We don't need to adjust the iterator, just the break condition.
                # The loop condition `if q > limit: break` uses the OLD limit.
                # We should update the check.
        
        # After the loop, if val > 1:
        # It has no prime factor <= min(p, sqrt(original_val)) that divided it?
        # Wait, the logic above is slightly flawed with dynamic limit.
        
        # Let's restart the logic for correctness:
        # 1. Divide by all primes q <= min(p, sqrt(val_current)).
        # 2. If val_current becomes 1, return True.
        # 3. If val_current > 1:
        #    - If val_current <= p, return True (factors are <= val_current <= p).
        #    - If val_current > p, it must be prime (no factors <= sqrt(val_current) found).
        #      So return False.
        
        # Correct implementation:
        val = val # reset
        limit = min(p, int(math.isqrt(val)))
        for q in primes:
            if q > limit:
                break
            if val % q == 0:
                while val % q == 0:
                    val //= q
                if val == 1:
                    return True
                # Update limit for next iteration
                limit = min(p, int(math.isqrt(val)))
        
        if val == 1:
            return True
        if val <= p:
            return True
        return False

    # ---------- Search ----------
    best_k = 0
    best_p = 2
    
    # Start time
    start_time = time.time()
    TIME_LIMIT = 110 # Leave 10s buffer
    
    # We search primes p.
    # We can't check all primes up to 5e6 if it takes too long.
    # We will check primes in increasing order.
    # If we find a good k, we might want to continue to see if we can do better.
    
    # To speed up, we can skip primes where p^2+1 is likely to have a large factor.
    # But we just check them.
    
    for p in primes:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        p2 = p * p
        k = 0
        for offset in range(1, 201):
            val = p2 + offset
            if is_p_smooth(val, p, primes):
                k = offset
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
