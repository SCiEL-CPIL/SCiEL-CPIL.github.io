import math

def construct():
    N = 123457
    
    # Step 1: Precompute the largest prime factor for all numbers up to N
    # lpf[i] will store the largest prime factor of i
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    # Sieve-like approach to find largest prime factors
    # We iterate through numbers and update multiples
    # A more efficient way: for each prime p, update all multiples
    # But a simple loop checking divisibility is O(N^2) which might be slow for 123k?
    # 123k^2 is ~1.5 * 10^10, which is too slow in Python.
    # We need a linear sieve or modified sieve.
    
    # Modified Sieve of Eratosthenes for Largest Prime Factor
    # Initialize lpf with 0
    # For i from 2 to N:
    #   if lpf[i] == 0: i is prime
    #      for j from i to N step i:
    #         lpf[j] = i  # Since we go i increasing, the last assignment is the largest prime factor
    
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is a prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # Step 2: Search for the pair (a, b) with a+b=N minimizing max(P(a), P(b))
    best_max_p = float('inf')
    best_a = 1
    best_b = N - 1
    
    # We only need to check up to N//2 because the pair (a, b) is the same as (b, a)
    for a in range(1, N // 2 + 1):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
            
    return [best_a, best_b]
