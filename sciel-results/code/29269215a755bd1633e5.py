import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_small(n, primes):
    """Compute phi(n) using precomputed primes up to sqrt(n)."""
    if n < 2:
        return n
    result = n
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            result -= result // p
            while temp % p == 0:
                temp //= p
    if temp > 1:
        result -= result // temp
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s safety margin
    
    # Precompute primes up to 10^6
    # sqrt(10^12) = 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    def check_run(s, k):
        """Check if phi(s), ..., phi(s+k-1) are pairwise distinct."""
        seen = set()
        for i in range(k):
            n = s + i
            if n < 1 or n + k - 1 > 10**12:
                return False
            v = phi_small(n, primes)
            if v in seen:
                return False
            seen.add(v)
        return True

    best_k = 1
    best_s = 1
    
    # Priority: Maximize k.
    # We search for k=200 first.
    # If found, return immediately.
    # If not found in budget, return best k < 200 found.
    
    # Strategy:
    # 1. Scan a small range near 10^12 (where values are large, collisions less likely? 
    #    Actually, distinctness depends on the map n->phi(n). 
    #    For large n, phi(n) ~ n. The range of phi values is [phi(s), phi(s+k)].
    #    If the range is small compared to the magnitude, distinctness is more likely?
    #    No, phi is not linear. But locally it varies.
    #    Random sampling is the most robust way to find a 'good' window.)
    
    # Generate candidates
    # We will test candidates in batches.
    # Each test of k=200 takes ~0.0001s? 200 phis * 1us = 0.2ms.
    # So we can do ~5000 tests per second.
    # In 100s, we can do 500,000 tests.
    
    # Deterministic scan first for reproducibility and to catch easy cases
    # Scan near 10^12
    start_scan = 10**12 - 200
    end_scan = 10**12 - 100000  # Scan 100k positions
    # This might be too slow if phi is slow.
    # Let's mix random and structured.
    
    random.seed(42)
    
    # List of start positions to test
    # We want to test k=200 primarily.
    
    # 1. Small range scan (fast, high density of potential hits if any)
    # Actually, small numbers have small phi, more collisions.
    # Large numbers have larger phi, but still even.
    
    # Let's prioritize random sampling in the high range [10^11, 10^12]
    # and also check a few specific structured regions.
    
    candidates_to_test = []
    
    # Structured: Near 10^12
    for i in range(0, 5000, 10):
        candidates_to_test.append(10**12 - 199 - i)
        
    # Random in [1, 10^12]
    for _ in range(50000):
        s = random.randint(1, 10**12 - 199)
        candidates_to_test.append(s)
        
    # Random in [10^11, 10^12] (Higher density of "good" candidates?)
    for _ in range(50000):
        s = random.randint(10**11, 10**12 - 199)
        candidates_to_test.append(s)

    # Shuffle to mix strategies
    random.shuffle(candidates_to_test)
    
    # Track best found
    best_k_found = 1
    best_s_found = 1
    
    # We only really care about k=200. If we find it, we win big.
    # If we don't, we might find k=199, etc.
    # But checking k=199 is just as fast as k=200 (slightly less work).
    # So we can just check k=200. If it fails, we don't automatically know k=199 works at that s.
    # We would need to check k=199 separately or re-check.
    # Given the probability of a run of 200 is low, finding one is the main prize.
    # If we fail to find 200, we can try to find 199.
    
    # Optimization: If we find a run of length 200, return it.
    # If time runs out, return the best found.
    # To find best found < 200, we'd need to check smaller k.
    # But if we can't find 200, finding 199 is also hard.
    # The seed probably found a small k.
    # So if we find ANY valid k > seed_k, we win.
    # Let's assume seed_k is small (e.g., 5-10).
    # So finding a run of 20 is a win. Finding 100 is a bigger win.
    # Finding 200 is the max win.
    
    # Revised Strategy:
    # Test k=200. If found, return.
    # If not, test k=100. If found, return (better than small k).
    # If not, test k=50.
    # etc.
    # This is inefficient if we keep failing.
    
    # Better:
    # For each candidate s:
    #   Check k=200. If valid, return.
    #   (Optionally check k=199, 198... but this doubles/triples time)
    #   If we just check 200, and fail, we don't know if 199 is valid.
    #   However, if 200 fails, it's likely 199 also fails or is close.
    #   Actually, if 200 fails due to a collision at position j, 199 might still be valid if the collision was at j=199? No, if j=199, 199-length run excludes the last element, so it might be valid.
    #   
    # Let's stick to checking k=200 primarily. 
    # If we find a run of 200, score is max.
    # If we don't, we might score 0 or low.
    # To be safe, we should record the max k found for any s.
    
    # Efficient way to find max k for a given s:
    # Compute phi for s..s+199. 
    # Then check suffixes/prefixes for distinctness?
    # This is complex.
    
    # Let's assume finding 200 is the goal.
    # If we can't find 200, we fall back to a known small run.
    # Known small run: s=1, k=1 (phi=1).
    # s=2, k=2? phi(2)=1, phi(3)=2. Distinct. k=2.
    # s=1, k=3? phi(1)=1, phi(2)=1. No.
    # s=2, k=3? phi(2)=1, phi(3)=2, phi(4)=2. No.
    # s=3, k=3? phi(3)=2, phi(4)=2. No.
    # s=5, k=4? 4,2,6,4. No.
    
    # I will implement a search for k=200.
    # If not found, I will do a quick search for k=100, 50, etc.
    
    for s in candidates_to_test:
        if time.time() > deadline:
            break
            
        if s < 1 or s + 199 > 10**12:
            continue
            
        if check_run(s, 200):
            return {'start': s, 'length': 200}
            
    # If k=200 not found, try smaller k values with new random candidates
    # This is a fallback to ensure we don't return 1 if we can find 10.
    target_ks = [100, 50, 20, 10, 5]
    
    for k in target_ks:
        if time.time() > deadline - 5:
            break
        # Sample 1000 candidates for this k
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            s = random.randint(1, 10**12 - k)
            if s < 1:
                continue
            if check_run(s, k):
                return {'start': s, 'length': k}
                
    # Fallback: Return a small valid run
    # s=2, k=2 is valid (1, 2)
    return {'start': 2, 'length': 2}
