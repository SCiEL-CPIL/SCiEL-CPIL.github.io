"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    N = 1000000
    window = 24
    max_val = N + window  # we need up to N+24
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Sieve: for each number, compute its "small-prime factor"
    # i.e., the product of all prime factors < 24 (with multiplicity)
    # We can do this by dividing out small primes from each number
    # More efficient: for each small prime p, for each multiple of p, divide out p
    
    # Initialize: spf[i] = i (the small-prime factor part)
    # Actually, we want: for number x, spf_part[x] = product of p^a for p in small_primes where p^a || x
    # Equivalently: spf_part[x] = x / (x with all small-prime factors removed)
    # = x / (x with all factors >= 24 removed)
    
    # Better approach: start with an array of 1s, and for each small prime p,
    # for each multiple m of p, multiply spf_part[m] by p while m is divisible by p
    # But that's O(N * sum(1/p)) which is fine.
    
    # Even better: for each number x, spf_part[x] = x with all prime factors >= 24 removed
    # We can compute this as: start with x, divide by each small prime as many times as possible
    # But that's O(N * 9) which is fine for N=1000024
    
    # Most efficient: use a sieve-like approach
    # spf_part[i] = i initially, then for each small prime p, divide out p from all multiples
    # Wait, that gives us the "large" part. We want the "small" part.
    
    # Let me think again. For number x:
    # x = (small part) * (large part)
    # where small part = product of p^a for p < 24
    # large part = product of q^b for q >= 24
    # We want the small part.
    
    # Approach: spf_part = np.arange(1, max_val+1)  # start with x itself
    # For each small prime p:
    #   for each multiple m of p:
    #     while spf_part[m] % p == 0:
    #       spf_part[m] //= p  # NO, this removes p from the small part, wrong direction
    
    # Correct approach:
    # large_part[i] = i initially
    # For each small prime p:
    #   for each multiple m of p:
    #     while large_part[m] % p == 0:
    #       large_part[m] //= p
    # Then spf_part[i] = i // large_part[i]
    
    # This is O(N * sum over small primes of (N/p) * log_p(N)) which is manageable.
    
    large_part = np.arange(1, max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        for m in range(p, max_val + 1, p):
            while large_part[m] % p == 0:
                large_part[m] //= p
    
    # Now spf_part[i] = i // large_part[i]
    spf_part = np.arange(1, max_val + 1, dtype=np.int64) // large_part
    
    # Now for each window starting at n (n+1 to n+24), count distinct spf_part values
    # We want to minimize this count.
    
    # Use a sliding window approach with a frequency dictionary
    # But for speed, let's just iterate and use a set for each window
    # That's O(N * 24) = O(24 * 1e6) = 24e6 operations, which might be slow in Python.
    
    # Better: use numpy and a smarter approach, or just do it in a loop with early termination.
    
    # Let's do a sliding window with a Counter-like approach using a dict.
    # For window starting at n: elements are spf_part[n+1], ..., spf_part[n+24]
    
    best_n = 1000
    best_count = 999999
    
    # Initialize the first window: n=1000, elements at indices 1001..1024
    from collections import defaultdict
    
    freq = defaultdict(int)
    for i in range(1001, 1001 + 24):
        freq[spf_part[i]] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
    
    # Slide the window
    for n in range(1001, N + 1):
        # Remove element at position n (which was spf_part[n])
        # Add element at position n + 24 (which is spf_part[n + 24])
        # Wait, window for n is [n+1, n+24]
        # Window for n-1 is [n, n+23]
        # So when sliding from n-1 to n: remove spf_part[n], add spf_part[n+24]
        
        if time.time() > deadline:
            break
        
        # Remove spf_part[n]
        val_remove = spf_part[n]
        freq[val_remove] -= 1
        if freq[val_remove] == 0:
            del freq[val_remove]
        
        # Add spf_part[n + 24]
        val_add = spf_part[n + 24]
        freq[val_add] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
