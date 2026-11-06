"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    WINDOW = 24
    MAX_VAL = MAX_N + WINDOW  # 1000024
    
    # Precompute the smooth part (part composed of primes < 24) for each number
    # smooth_part[m] = largest divisor of m using only primes in `primes`
    smooth = [1] * (MAX_VAL + 1)
    
    for p in primes:
        # For each multiple of p, divide out all powers of p
        for m in range(p, MAX_VAL + 1, p):
            x = m
            while x % p == 0:
                x //= p
            # The smooth part contribution from p is m // x
            # But we need the full smooth part, so we multiply contributions
            # Better approach: for each m, compute smooth part directly
            pass
    
    # More efficient: compute smooth part for each number
    # smooth_part[m] = m / (part of m with primes >= 29)
    # i.e., remove all prime factors >= 29
    smooth_part = list(range(MAX_VAL + 1))  # initialize
    
    # For each prime >= 29, remove its contribution from all multiples
    # Actually, let's just compute it directly:
    # smooth_part[m] = product of p^a for p in primes where p^a || m
    
    # Initialize
    smooth_part = [1] * (MAX_VAL + 1)
    for p in primes:
        for m in range(p, MAX_VAL + 1, p):
            x = m
            power = 1
            while x % p == 0:
                power *= p
                x //= p
            smooth_part[m] *= power
    
    # Now use sliding window to find min distinct smooth parts
    # For window starting at n (n in [1000, 1000000]), look at smooth_part[n+1..n+24]
    
    best_n = 1000
    best_count = float('inf')
    
    # Sliding window with frequency counter
    freq = defaultdict(int)
    
    # Initialize first window: n=1000, elements are smooth_part[1001..1024]
    for i in range(1001, 1001 + WINDOW):
        freq[smooth_part[i]] += 1
    
    distinct_count = len(freq)
    if distinct_count < best_count:
        best_count = distinct_count
        best_n = 1000
    
    # Slide the window
    for n in range(1001, MAX_N + 1):
        # Remove smooth_part[n] (which was at position n in the previous window)
        # Previous window was [n, n+23], new window is [n+1, n+24]
        # Wait, let me re-index.
        # Window for n: elements are smooth_part[n+1], ..., smooth_part[n+24]
        # Window for n-1: elements are smooth_part[n], ..., smooth_part[n+23]
        # So when moving from n-1 to n: remove smooth_part[n], add smooth_part[n+24]
        
        # Actually my loop starts at n=1001, and the initial window was for n=1000
        # Window for n=1000: smooth_part[1001..1024]
        # Window for n=1001: smooth_part[1002..1025]
        # Remove: smooth_part[1001], Add: smooth_part[1025]
        
        remove_val = smooth_part[n]  # smooth_part[1001] when n=1001
        add_val = smooth_part[n + WINDOW - 1]  # smooth_part[1024] when n=1001
        
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            del freq[remove_val]
        
        freq[add_val] += 1
        
        distinct_count = len(freq)
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
