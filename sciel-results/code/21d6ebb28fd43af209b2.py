"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _largest_prime_factor(n):
    """Compute the largest prime factor of n via trial division."""
    if n <= 1:
        return 1
    largest = 1
    # Check factor 2
    while n % 2 == 0:
        largest = 2
        n //= 2
    # Check odd factors up to sqrt(n)
    f = 3
    while f * f <= n:
        while n % f == 0:
            largest = f
            n //= f
        f += 2
    if n > 1:
        largest = n
    return largest

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # We want to minimize the largest prime factor of n*(n+1).
    # The largest prime factor of a product is max(lpf(n), lpf(n+1)).
    # So we want to find n such that both n and n+1 have small largest prime factors.
    # This is equivalent to finding consecutive integers that are both "smooth"
    # (have no large prime factors).
    
    # Strategy: Precompute largest prime factors for all numbers in [10000, 1000000]
    # using a sieve-like approach, then scan for the minimum max(lpf(n), lpf(n+1)).
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Sieve for largest prime factor
    # lpf[i] will store the largest prime factor of i
    size = N_MAX + 1
    lpf = [0] * size
    
    # Initialize: for each number, its largest prime factor is itself initially
    for i in range(2, size):
        lpf[i] = i
    
    # Sieve: for each prime p, mark all multiples of p
    # We only need to process primes up to sqrt(N_MAX) for the sieve to work efficiently
    # Actually, we need to process all primes, but we can optimize by only iterating
    # up to sqrt for the outer loop and handling the rest.
    
    # More efficient approach: iterate i from 2 to sqrt(N_MAX), and if lpf[i] == i (it's prime),
    # then for all multiples j of i, set lpf[j] = max(lpf[j], i).
    # But actually, since we're going in increasing order, and we want the LARGEST prime factor,
    # we should process primes in increasing order and update multiples.
    # Wait, if we process in increasing order, the last prime that divides a number is its largest.
    # So: for each i from 2 to N_MAX, if lpf[i] == i (meaning i is prime), then for all multiples
    # j = i, 2i, 3i, ... <= N_MAX, set lpf[j] = i. Since we go in increasing i, the last assignment
    # to lpf[j] will be from the largest prime factor of j.
    
    # But this is O(N log N) which might be too slow in Python for N=1000000.
    # Let's use a different approach: for each number, we can factor it, but that's also slow.
    
    # Alternative: Use a modified sieve where we only track the largest prime factor.
    # For i from 2 to N_MAX:
    #   if lpf[i] == 0 (not set, meaning i is prime), then for j in range(i, N_MAX+1, i):
    #     lpf[j] = i  (since we go in increasing order, the last prime that hits j is its largest)
    
    # This is O(N log log N) which is fast enough.
    
    # Actually, let me reconsider. The standard sieve for smallest prime factor is:
    # for i in range(2, N+1):
    #   if spf[i] == 0:  # i is prime
    #     for j in range(i, N+1, i):
    #       if spf[j] == 0:
    #         spf[j] = i
    # 
    # For largest prime factor, we can do:
    # lpf = [0] * (N+1)
    # for i in range(2, N+1):
    #   if lpf[i] == 0:  # i is prime
    #     for j in range(i, N+1, i):
    #       lpf[j] = i  # keep overwriting, so the last (largest) prime factor wins
    
    # But wait, this sets lpf[j] to the largest prime factor because we iterate i in increasing order,
    # and for each prime i, we set lpf[j] = i for all multiples j. The last prime i that divides j
    # is the largest prime factor of j.
    
    # Let's implement this.
    
    lpf = [0] * size
    for i in range(2, size):
        if lpf[i] == 0:  # i is prime
            for j in range(i, size, i):
                lpf[j] = i
    
    # Now lpf[n] contains the largest prime factor of n for all n in [2, N_MAX]
    # For n=0 and n=1, lpf is 0, but we don't care about those.
    
    # Now scan for the best n in [N_MIN, N_MAX-1] (since we need n+1 <= N_MAX)
    best_n = N_MIN
    best_score = float('inf')
    
    # Check time before the scan
    if time.time() > deadline - 5:
        return best_n
    
    for n in range(N_MIN, N_MAX):
        # Largest prime factor of n*(n+1) is max(lpf[n], lpf[n+1])
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            # If we found a very good score, we might be able to stop early
            # But let's finish the scan since it's fast
        # Check time periodically
        if n % 100000 == 0 and time.time() > deadline - 5:
            break
    
    return best_n

# EVOLVE-BLOCK-END
