import math
import time
import random

def construct():
    deadline = time.time() + 113.0

    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Generate primes up to a reasonable limit
    # 10^7 is about 660k primes. Sieve takes ~0.5-1s.
    # 10^8 is about 5.7M primes. Sieve takes ~5-10s and 100MB RAM.
    # Let's try 10^7 first to be safe on memory/time, maybe increase if fast.
    SIEVE_LIMIT = 10**7
    primes = sieve(SIEVE_LIMIT)
    
    def check_run_segmented(p, primes_list):
        K = 200
        p_sq = p * p
        # We need to track the remaining part of p^2 + i
        # Initialize with the numbers themselves
        # Using a list of ints
        rem = [p_sq + i for i in range(1, K + 1)]
        
        # Sieve rem with primes <= p
        # Optimization: if q*q > max(rem), we can stop?
        # No, because rem changes.
        # But we can stop if all are 1.
        
        # Iterate through primes
        # We only need primes <= p
        # Since primes_list is sorted, we can break when q > p
        
        for q in primes_list:
            if q > p:
                break
            
            # Find first multiple of q in [p_sq+1, p_sq+K]
            # start = ceil((p_sq+1)/q)*q
            start_val = ((p_sq + 1 + q - 1) // q) * q
            start_idx = start_val - (p_sq + 1)
            
            if start_idx >= K:
                continue
                
            # Mark multiples
            # Check if all are 1 to break early?
            # It's cheap to check all rem == 1 occasionally?
            # Or just proceed.
            
            for j in range(start_idx, K, q):
                if rem[j] % q == 0:
                    while rem[j] % q == 0:
                        rem[j] //= q
                        if rem[j] == 1:
                            break
                            # Note: if rem[j] becomes 1, it stays 1.
                            # We can skip further divisions for this j.
                            # But the loop `while rem[j] % q == 0` handles it.
                            # Actually, if rem[j] == 1, 1 % q != 0, so loop ends.
        
        # Determine max k
        k = 0
        for i in range(K):
            if rem[i] == 1:
                k = i + 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 2
    
    # Check small primes first (quick wins)
    for p in primes:
        if time.time() > deadline:
            break
        
        # We can skip if p is too large?
        # The check is O(pi(p)).
        # For p=10^7, pi(p) ~ 660k.
        # 660k * (200/q) ops.
        # This is fast.
        
        k = check_run_segmented(p, primes)
        if k > best_k:
            best_k = k
            best_p = p
            
        # If we found a very long run, maybe stop?
        # Max k is 200.
        if best_k >= 200:
            break
            
    return {'p': best_p, 'k': best_k}
