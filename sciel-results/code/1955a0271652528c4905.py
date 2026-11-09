"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import heapq

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 123457
    
    def generate_smooth_numbers(N, primes):
        """Generate all numbers <= N with prime factors in 'primes' using efficient BFS."""
        seen = set([1])
        queue = [1]
        idx = 0
        while idx < len(queue):
            current = queue[idx]
            idx += 1
            for p in primes:
                nxt = current * p
                if nxt <= N and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen
    
    # Try to find a decomposition where both numbers are k-smooth for small k.
    # If a and b are k-smooth, then P(a) <= k and P(b) <= k, so max(P(a), P(b)) <= k.
    # We want to minimize this value.
    
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We check increasing sets of primes.
    for i in range(len(primes_list)):
        if time.time() > DEADLINE:
            break
            
        current_primes = primes_list[:i+1]
        smooth_numbers = generate_smooth_numbers(N, current_primes)
        
        # Check if any x in smooth_numbers has N-x also in smooth_numbers
        for x in smooth_numbers:
            y = N - x
            if 1 <= y <= N and y in smooth_numbers:
                # Found a pair where both are smooth with respect to current_primes
                # max(P(x), P(y)) <= max(current_primes) = current_primes[-1]
                return [x, y]
    
    # Fallback to the sieve method if no smooth pair found
    lpf = np.ones(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            lpf[i::i] = i
            
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    pa = lpf[a]
    pb = lpf[b]
    m = np.maximum(pa, pb)
    idx = np.argmin(m)
    
    best_a = int(a[idx])
    best_b = int(b[idx])
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
