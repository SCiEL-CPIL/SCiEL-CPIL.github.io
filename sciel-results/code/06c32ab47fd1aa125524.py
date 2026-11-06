"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    
    # We need small parts for numbers up to MAX_N + WINDOW_SIZE
    M = MAX_N + WINDOW_SIZE + 10
    
    # Precompute small parts for all numbers up to M
    # sp[x] will hold the product of prime factors < 24 of x
    # Use array for better cache performance
    sp = array('I', [1]) * (M + 1)
    rem = array('I', range(M + 1))
    
    # Sieve-like computation for small parts
    for p in primes:
        for j in range(p, M + 1, p):
            # Extract all factors of p from rem[j]
            while rem[j] % p == 0:
                sp[j] *= p
                rem[j] //= p
    
    # Sliding window to find min distinct small parts
    # Window covers n+1 to n+24
    # We start with n = MIN_N
    # Initial window values: sp[MIN_N+1] ... sp[MIN_N+24]
    
    counts = {}
    distinct = 0
    
    # Initialize counts for the first window
    sp_local = sp
    MIN_N_local = MIN_N
    WINDOW_SIZE_local = WINDOW_SIZE
    
    for i in range(1, WINDOW_SIZE_local + 1):
        v = sp_local[MIN_N_local + i]
        c = counts.get(v, 0)
        if c == 0:
            distinct += 1
        counts[v] = c + 1
        
    best_n = MIN_N
    best_distinct = distinct
    
    # Slide the window
    # n goes from MIN_N to MAX_N
    # When moving from n to n+1:
    # Remove sp[n+1]
    # Add sp[n+1+WINDOW_SIZE] = sp[n+25]
    
    # Inline the add/remove logic for performance
    # We check time every 50000 iterations to avoid syscall overhead
    
    for n in range(MIN_N, MAX_N + 1):
        # Update best
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = n
            
        # Check time every 50000 steps
        if (n - MIN_N) % 50000 == 0:
            if time.time() > deadline:
                break
        
        # Prepare for next n (n+1)
        if n < MAX_N:
            # Remove the leftmost element of the current window: sp[n+1]
            v_out = sp_local[n + 1]
            c_out = counts[v_out]
            if c_out == 1:
                distinct -= 1
                del counts[v_out]
            else:
                counts[v_out] = c_out - 1
            
            # Add the new rightmost element: sp[n+1+WINDOW_SIZE]
            v_in = sp_local[n + 1 + WINDOW_SIZE_local]
            c_in = counts.get(v_in, 0)
            if c_in == 0:
                distinct += 1
            counts[v_in] = c_in + 1
            
    return best_n
# EVOLVE-BLOCK-END
