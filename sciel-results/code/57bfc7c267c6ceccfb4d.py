import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s
    
    N = 1000000
    # We need numbers from 1001 to 1000024
    # Compute 23-smooth part for each
    
    # Primes >= 29 up to sqrt(1000024) ~ 1000
    # Actually, we need primes >= 29 up to 1000024
    # But for the sieve, we only need primes >= 29 up to 1000024
    
    # Let's precompute primes >= 29 up to 1000024
    # Actually, for dividing out, we need all primes >= 29
    
    # Better: use a sieve to compute the 23-smooth part directly
    # smooth[m] = m with all prime factors >= 29 removed
    
    max_val = 1000024
    # Initialize smooth[m] = m for all m
    smooth = list(range(max_val + 1))
    
    # Sieve: for each prime p >= 29, divide out p from all multiples
    # First, find all primes >= 29 up to max_val
    # Use a simple sieve to find primes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # For each prime p >= 29, divide out all factors of p from smooth[multiples]
    for p in range(29, max_val + 1):
        if is_prime[p]:
            for m in range(p, max_val + 1, p):
                while smooth[m] % p == 0:
                    smooth[m] //= p
    
    # Now smooth[m] is the 23-smooth part of m
    # For each window [n+1, n+24], count distinct smooth values
    # n ranges from 1000 to 1000000
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window approach
    # For window starting at n (i.e., numbers n+1 to n+24)
    # Use a frequency counter
    
    from collections import Counter
    
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = Counter()
    for i in range(1, 25):
        freq[smooth[1000 + i]] += 1
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide window
    for n in range(1001, N + 1):
        if time.time() > deadline:
            break
        # Remove n (which was n-1+1 = n, the leftmost)
        # Window was [n, n+23] for n-1, now [n+1, n+24] for n
        # Wait, let me re-index.
        # For n-1: window is [(n-1)+1, (n-1)+24] = [n, n+23]
        # For n: window is [n+1, n+24]
        # So remove smooth[n], add smooth[n+24]
        
        val_remove = smooth[n]
        freq[val_remove] -= 1
        if freq[val_remove] == 0:
            del freq[val_remove]
        
        val_add = smooth[n + 24]
        freq[val_add] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
