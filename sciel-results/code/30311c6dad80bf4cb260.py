import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # 1. Generate primes up to 10^7 for the segmented sieve
    # This allows us to sieve the interval [p^2+1, p^2+200] efficiently.
    # For p up to 10^7, p^2 is up to 10^14.
    # Any prime factor q of n in [p^2+1, p^2+200] that is > 10^7 
    # must be the largest prime factor (LPF).
    # If LPF > p, then n is not p-smooth.
    # If LPF <= p, then n is p-smooth.
    # Wait, if n has a factor q > 10^7, and q <= p, then n is p-smooth.
    # But we need to know if q > p.
    # If we sieve with all primes up to 10^7, the remaining value R for each n 
    # will be either 1 (fully factored) or a prime > 10^7 (or a product of primes > 10^7, 
    # but since n < 10^14, it can have at most one such factor).
    # So R is either 1 or a prime.
    # If R == 1, max factor was <= 10^7.
    # If R > 1, R is a prime factor.
    # We need to check if all prime factors are <= p.
    # During sieving, we can track the largest prime factor found so far.
    # Let max_pf[i] be the largest prime factor found for n_i.
    # After sieving with primes <= 10^7, if R[i] > 1, then max_pf[i] = max(max_pf[i], R[i]).
    # Then n_i is p-smooth iff max_pf[i] <= p.

    limit_primes = 10**7
    sieve = [True] * (limit_primes + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_primes)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # We only need primes up to 10^7 for the sieve.
    # For p > 10^7, the logic changes slightly or we need larger primes.
    # Let's focus on p <= 10^7 first, as the current program does.
    # The current program finds k=7 (2.3333/3) for some p.
    # Let's see if we can find larger k for larger p.
    # The probability of a run of length k is roughly (1/ln p)^k? No.
    # The density of p-smooth numbers is rho(log n / log p).
    # For n ~ p^2, log n ~ 2 log p. So rho(2).
    # rho(2) ~ 0.3.
    # So the expected run length is 1/(1-0.3) ~ 1.4? No, geometric distribution.
    # The probability of a specific position being p-smooth is ~0.3.
    # The probability of a run of length k is ~0.3^k.
    # For k=7, 0.3^7 ~ 2e-4.
    # Number of primes up to 10^7 is ~664,000.
    # Expected number of runs of length 7 is 664000 * 200 * 2e-4 ~ 26000.
    # So we should find many runs of length 7.
    # The current score is 2.3333, which corresponds to k=7 if seed=3.
    # Wait, score = value/seed. If value=k, and seed=3, score=7/3=2.333.
    # So the current program finds k=7.
    # Can we find k=8 or more?
    # Probability of k=8 is 0.3^8 ~ 6.5e-5.
    # Expected runs: 664000 * 200 * 6.5e-5 ~ 8600.
    # So k=8 is likely.
    # Why did the current program only find 7?
    # Maybe it didn't search enough primes or was too slow.
    # My batched sieve should be much faster, allowing me to search all primes up to 10^7.

    def find_max_k_batched(p):
        base = p * p
        L = 200
        # We need to check n = base + 1, ..., base + L
        # Initialize max_pf for each n
        max_pf = [0] * (L + 1)
        # We will sieve the interval [base+1, base+L] with primes q <= 10^7
        # For each prime q, find the first multiple of q in the interval.
        # Let start = ceil((base+1)/q) * q
        # If start < base+1, start += q
        # For each multiple m = start, start+q, ..., <= base+L:
        #   idx = m - base
        #   while m % q == 0:
        #       m //= q
        #       max_pf[idx] = max(max_pf[idx], q)
        
        # Optimization: Only sieve with primes q <= min(p, 10^7)
        # If q > p, we don't need to update max_pf with q, because if q divides n,
        # and q > p, then n is NOT p-smooth.
        # But we can just check at the end: if max_pf[idx] > p, not smooth.
        # If we don't sieve with q > p, then max_pf[idx] might be smaller than q.
        # But if q > p and q divides n, then n has a factor > p.
        # So we need to detect this.
        # If we only sieve with q <= p, then any remaining factor R > 1 must be > p?
        # No, R could be a prime q' where p < q' <= 10^7.
        # If we stop sieving at p, then R might contain factors in (p, 10^7].
        # So we should sieve with all primes q <= 10^7.
        # Then max_pf[idx] will be the largest prime factor <= 10^7.
        # The remaining value R[idx] will be 1 or a prime > 10^7.
        # If R[idx] > 1, then max_pf[idx] = max(max_pf[idx], R[idx]).
        # Then check if max_pf[idx] <= p.

        # Let's implement this.
        # To speed up, we can precompute the starting index for each prime.
        # But doing it on the fly is fine for L=200.
        
        for q in primes:
            if q > 10**7:
                break
            # Find first multiple of q in [base+1, base+L]
            # start = ((base + 1 + q - 1) // q) * q
            start = ((base + 1 + q - 1) // q) * q
            if start < base + 1:
                start += q
            if start > base + L:
                continue
            
            # Sieve
            for m in range(start, base + L + 1, q):
                idx = m - base
                # Remove all factors of q
                while m % q == 0:
                    m //= q
                max_pf[idx] = max(max_pf[idx], q)
        
        # Now check the remainder
        # We need to know the remainder R for each n.
        # We didn't store it. Let's recompute or store it.
        # Storing it is easier.
        # Let's modify the loop to store the remainder.
        
        # Actually, we can just check if max_pf[idx] <= p.
        # But what if there is a factor > 10^7?
        # If n has a factor > 10^7, it must be prime (since n < 10^14).
        # Let R be the remainder after dividing by all primes <= 10^7.
        # If R > 1, then R is a prime factor.
        # We need to include R in max_pf.
        
        # Let's redo with storing remainder.
        rem = [0] * (L + 1)
        for i in range(1, L + 1):
            rem[i] = base + i
            max_pf[i] = 0
        
        for q in primes:
            if q > 10**7:
                break
            start = ((base + 1 + q - 1) // q) * q
            if start < base + 1:
                start += q
            if start > base + L:
                continue
            
            for m in range(start, base + L + 1, q):
                idx = m - base
                while rem[idx] % q == 0:
                    rem[idx] //= q
                max_pf[idx] = max(max_pf[idx], q)
        
        # Final check
        k = 0
        for i in range(1, L + 1):
            if rem[i] > 1:
                max_pf[i] = max(max_pf[i], rem[i])
            
            if max_pf[i] <= p:
                k += 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 3
    
    # Search primes up to 10^7
    # We can process them in batches or one by one.
    # One by one is fine, as the sieve is fast.
    for p in primes:
        if p > 10**7:
            break
        if time.time() > deadline:
            break
        
        k = find_max_k_batched(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
