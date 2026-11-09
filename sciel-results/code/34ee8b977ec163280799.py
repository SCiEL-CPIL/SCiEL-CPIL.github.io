"""
Strategy: Hardcoded Optimal Solution
The task asks to maximize k for a prime p <= 10^9 such that p^2+1 ... p^2+k
have no prime factors > p.
The seed program searches p < 2000.
A more effective strategy, given the 150s time limit, is to identify a
specific (p, k) pair that yields a high k, verify its validity mentally or
via offline computation, and hardcode it. 
Based on known results for "smooth numbers" in arithmetic progressions 
or simply by assuming a moderate p allows for a significantly larger k than 
the seed's limited search range, we select a known good candidate.

For example, p=3 gives k=2 (9+1=10=2*5, 9+2=11 prime>3).
Let's look at a slightly larger prime. 
Actually, the seed found best=(0,2) in its limited range? No, seed says best=(0,2) initially, 
then updates. 
Let's assume a prime like p=101 or p=1009 might allow larger k.
However, without running the code, I will implement a robust runtime search 
that is MUCH faster than the seed, using a simple primality test and 
trial division up to sqrt(n) but with early termination and a time guard.
Wait, the prompt says "COMPUTE BUDGET: ... return the best result found".
The seed is too slow because it uses `range(3, int(p**0.5)+1, 2)` for EVERY number in the sequence.
If I use a pre-computed list of small primes for trial division, I can check much larger p.

Revised Strategy:
1. Generate primes up to a limit (e.g., 10^5) using Sieve of Eratosthenes.
2. Iterate through these primes as candidates for p.
3. For each p, check k=1, 2, ... using the pre-computed small primes for trial division.
   If a number has a factor > p, stop.
   To be exact, if the number is not divisible by any prime <= p, it must have a factor > p (unless it is 1, but we are looking at p^2+1 >= 4).
   Actually, if n = p^2 + i, and we divide out all primes <= p, the remainder must be 1 for the condition to hold.
   If remainder > 1, it has a prime factor > p.
4. This is much faster than the seed's naive trial division.
5. Add a time guard to stop if > 140s.
"""

import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Sieve primes up to a limit.
    # We need to test p up to some limit. Let's try primes up to 100,000.
    # This allows checking p^2 + k for k up to 200.
    # The factorization of p^2+k requires checking divisors up to p.
    # If we have all primes up to 10^5, we can fully factor any number < (10^5)^2 = 10^10 
    # IF the number is smooth with respect to 10^5.
    # The condition is: all prime factors <= p.
    # So if p is in our sieve, we can check the condition exactly.
    
    limit = 100000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    best_k = 0
    best_p = 3
    
    # Iterate through candidate primes p
    # We can't check all primes up to 100,000 if the inner loop is slow.
    # Inner loop: for k in 1..200, check if p^2+k is p-smooth.
    # Checking p-smoothness: divide by all primes <= p.
    # For p=100,000, there are ~9,500 primes. 200 * 9500 = 1.9M operations per p.
    # If we check 1000 primes, that's 1.9B ops. Too slow in Python.
    
    # Optimization: 
    # We only need to check k up to 200.
    # We can stop checking k for a given p if k exceeds best_k.
    # Also, we can skip p if p^2+1 has a large factor.
    
    # Let's refine the search space.
    # The seed checked p < 2000.
    # Let's check p up to 10,000 first.
    # Primes up to 10,000: ~1229 primes.
    # For p=10,000, primes <= p are ~1229.
    # 200 * 1229 = 245,800 ops per p.
    # 1229 primes * 245,800 = 300M ops. Might be tight in 140s in Python.
    # Let's use a faster check: 
    # Instead of dividing by ALL primes <= p, we can just check if the number is 1 after dividing by all primes <= p.
    # Actually, we can break early if the remainder is 1.
    # Also, if the remainder is > 1, it's a fail.
    
    # Even better: 
    # We can just iterate p. 
    # To speed up, we can pre-calculate p^2.
    
    for p in primes:
        if p > limit:
            break
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        
        # We only care if k > best_k
        # If best_k is already high, we can skip smaller k checks? No, we must check sequentially.
        # But we can stop if k reaches best_k and we don't beat it? 
        # Actually, we want to maximize k. 
        # If we find k=10 for p=100, and best_k is 5, we update.
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            val = p_sq + i
            
            # Check if val has any prime factor > p
            # This is equivalent to: val is p-smooth.
            # We divide val by all primes <= p.
            # If the result is 1, it's p-smooth.
            # If the result is > 1, it has a factor > p (since we removed all factors <= p).
            
            temp = val
            # Optimization: only check primes <= min(p, sqrt(val))? 
            # No, we need to remove ALL factors <= p.
            # If a factor q <= p divides val, we remove it.
            # If after removing all q <= p, temp > 1, then temp is composed of primes > p.
            
            # We can speed this up by only iterating primes up to p.
            # But doing this for every i is slow.
            
            # Alternative: 
            # If best_k is small, we can afford to check more p.
            # If best_k is large, we should focus on promising p.
            
            # Let's just do the trial division.
            # To make it faster in Python, we'll break early if possible.
            # But we MUST remove all small factors to know if the remainder is 1.
            
            # Trick: 
            # If val is prime and val > p, fail.
            # If val is composite, check factors.
            
            # Let's use a helper that is as fast as possible.
            # We'll iterate primes from the list.
            # Since p is increasing, the list of primes <= p is a prefix of `primes`.
            
            failed = False
            t = val
            for q in primes:
                if q > p:
                    break
                if q * q > t:
                    # If t > 1 and q*q > t, then t is prime.
                    # If t > 1, then t is a prime factor.
                    # We need t <= p for success.
                    if t > 1 and t > p:
                        failed = True
                    break
                while t % q == 0:
                    t //= q
                if t == 1:
                    break
            
            if not failed and t == 1:
                k = i
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # Update deadline? No, fixed deadline.
            # Maybe we can stop early if best_k is 200?
            if best_k == 200:
                break

    return {'p': best_p, 'k': best_k}
