"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return np.array([i for i in range(limit + 1) if is_prime[i]], dtype=np.int64)

def compute_phi_window(s, k, primes):
    """
    Compute phi(s), phi(s+1), ..., phi(s+k-1).
    Uses a segmented sieve to find the smallest prime factor (SPF) for each number in the window,
    then performs trial division using the SPF and remaining primes.
    """
    # 1. Segmented Sieve for SPF
    # We need SPF for numbers in [s, s+k-1]
    # Primes up to sqrt(s+k-1) are sufficient to factorize any number in the range.
    # Since s+k-1 <= 10^12, sqrt is 10^6.
    
    spf = np.zeros(k, dtype=np.int64)
    max_val = s + k - 1
    
    # Mark SPF for each prime p
    for p in primes:
        if p * p > max_val:
            break
        
        # Find the first multiple of p in [s, s+k-1]
        first = ((s + p - 1) // p) * p
        
        # Mark spf for all multiples of p in the window
        # We only set spf if it hasn't been set yet (i.e., p is the smallest prime factor)
        # However, iterating over all multiples is O(k * pi(sqrt(N))).
        # A more efficient way is to iterate p and mark.
        for j in range(first, s + k, p):
            idx = j - s
            if spf[idx] == 0:
                spf[idx] = p

    # 2. Compute Phi for each number
    phis = np.zeros(k, dtype=np.int64)
    
    # Pre-compute prime squares for quick check? No, just iterate primes.
    # For each number n in the window:
    # phi(n) = n * prod(1 - 1/p) for all distinct prime factors p of n.
    
    # We can vectorize the initial multiplication, but the factorization loop is sequential per number.
    # However, k is small (<= 200), so a Python loop over k is fine if the inner factorization is fast.
    
    for i in range(k):
        n = s + i
        temp = n
        phi_val = n
        
        # Divide by small prime factors found in spf
        # Note: spf[i] gives the smallest prime factor.
        # We need to find ALL distinct prime factors.
        # We can use the spf array to guide us, but spf only gives the smallest.
        # Actually, the segmented sieve above sets spf[idx] to the smallest prime p such that p divides n.
        # To get all factors, we can just do trial division with the global primes list,
        # but stop when p*p > temp.
        
        # Optimization: Use spf to skip to the first factor?
        # If spf[i] is 0, n is prime (or 1).
        # If spf[i] > 0, it's the smallest prime factor.
        
        # Let's just use trial division with the primes list, which is fast enough for 200 numbers.
        # Primes are up to 10^6. There are ~78k primes.
        # Worst case: n is a product of two large primes. We iterate until sqrt(n).
        # But for n ~ 10^12, sqrt is 10^6.
        # Iterating 78k primes for 200 numbers is ~15M operations. This might be slow in Python.
        
        # Better approach: Use the spf array to extract factors.
        # The spf array in the segmented sieve only stores the SMALLEST prime factor.
        # We need to factorize completely.
        # Since we have the primes list, we can check divisibility.
        
        # Let's stick to the trial division but break early.
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                phi_val -= phi_val // p
                while temp % p == 0:
                    temp //= p
        
        if temp > 1:
            phi_val -= phi_val // temp
        
        phis[i] = phi_val

    return phis

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    # We want to maximize k.
    # It is extremely likely that k=200 is valid for s near 10^12.
    # Let's try to find a valid s for k=200.
    
    # Strategy:
    # 1. Try s = max_n - 200 + 1, max_n - 400 + 1, etc.
    # 2. If collision found, try smaller k? Or just move s?
    # Moving s is better because we want max k.
    
    # Let's scan backwards from max_n.
    # We can check multiple candidates in parallel? No, sequential is fine if fast.
    
    # Check function
    def check_distinct(s, k):
        if s < 1 or s + k - 1 > max_n:
            return False
        phis = compute_phi_window(s, k, primes)
        # Check for duplicates
        # Sorting is O(k log k). For k=200, this is very fast.
        # Or use a set.
        if len(set(phis)) != k:
            return False
        return True

    # Try k=200
    # We'll check a few positions.
    # If we find one, return it.
    
    # Start from the top.
    # s = max_n - 200 + 1
    # If that fails, try s = max_n - 400 + 1, etc.
    
    # Let's try 100 different starting positions for k=200.
    # If none work (highly unlikely), try k=199, etc.
    
    for offset in range(0, 1000, 10):
        s = max_n - 200 + 1 - offset
        if s < 1:
            break
        if time.time() > deadline - 10:
            break
        if check_distinct(s, 200):
            return {'start': s, 'length': 200}
            
    # If k=200 not found quickly, try smaller k with more candidates
    for k in [199, 198, 195, 190, 180, 150, 100, 50, 10]:
        if time.time() > deadline - 5:
            break
        for offset in range(0, 500, 5):
            s = max_n - k + 1 - offset
            if s < 1:
                break
            if check_distinct(s, k):
                return {'start': s, 'length': k}
                
    # Fallback
    return {'start': 1, 'length': 1}

# EVOLVE-BLOCK-END
