"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    N = 1000000
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute the bitmask of small-prime factors for each integer in [1, N+24]
    # bit i is set if primes[i] divides the number
    # We need to handle numbers up to N + 24 (since n can be up to N and we look at n+1..n+24)
    max_val = N + 24
    
    # Sieve-like approach: for each prime, mark all multiples
    # Use a list of integers where bit i indicates divisibility by primes[i]
    masks = [0] * (max_val + 1)
    
    for i, p in enumerate(primes):
        bit = 1 << i
        for multiple in range(p, max_val + 1, p):
            masks[multiple] |= bit
    
    # Now find the window [n+1, n+24] that minimizes the number of distinct prime factors
    # across all 24 numbers.
    # The "extracted factor" for n+i is the product of all small primes dividing n+i.
    # But the task says "MINIMIZE the number of distinct extracted factors".
    # Wait, let me re-read: "extract its complete factor composed of primes STRICTLY BELOW 24. MINIMIZE the number of distinct extracted factors."
    
    # So for each n+i, we extract the part of n+i that is composed of primes < 24.
    # That is, if n+i = 2^a * 3^b * 5^c * ... * 23^k * (co-prime part), the extracted factor is 2^a * 3^b * 5^c * ... * 23^k.
    # We want to minimize the number of DISTINCT such extracted factors across the 24 values.
    
    # Actually, re-reading more carefully: "extract its complete factor composed of primes STRICTLY BELOW 24"
    # This means for each n+i, we take the part that consists only of primes < 24.
    # So if n+i = 2^3 * 7 * 31, the extracted factor is 2^3 * 7 = 56.
    # We want to minimize the number of distinct such factors across the 24 consecutive integers.
    
    # So I need to compute the actual extracted factor value, not just the bitmask.
    # Let me recompute: for each number, compute the product of all prime powers where the prime is < 24.
    
    # Recompute with actual values
    extracted = [1] * (max_val + 1)
    for i, p in enumerate(primes):
        # For each multiple of p, divide out all powers of p and multiply into extracted
        # Actually, it's easier to compute the "small prime part" directly
        pass
    
    # Let me redo this properly
    # For each number m, extracted[m] = product of p^v where p < 24 and p^v || m
    # This is the part of m composed of primes < 24.
    
    # Initialize extracted as 1 for all
    extracted = [1] * (max_val + 1)
    
    for p in primes:
        p_power = p
        while p_power <= max_val:
            for multiple in range(p_power, max_val + 1, p_power):
                # If p_power divides multiple, then p divides multiple at least this many times
                # We need to multiply extracted[multiple] by p for each power
                extracted[multiple] *= p
            p_power *= p
    
    # Now extracted[m] is the complete factor of m composed of primes < 24.
    # For window starting at n (looking at n+1 to n+24), we want to minimize
    # the number of distinct values in {extracted[n+1], extracted[n+2], ..., extracted[n+24]}
    
    # Use a sliding window with a frequency map
    best_n = 1000
    best_count = float('inf')
    
    # Initial window: n=1000, so we look at 1001 to 1024
    freq = {}
    for i in range(1, 25):
        val = extracted[1000 + i]
        freq[val] = freq.get(val, 0) + 1
    
    current_distinct = len(freq)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, N + 1):
        # Remove extracted[n] (which was n+1 in the previous window... wait, let me be careful)
        # Previous window was for n-1: looking at (n-1)+1 to (n-1)+24 = n to n+23
        # Current window is for n: looking at n+1 to n+24
        # So we remove extracted[n] and add extracted[n+24]
        
        old_val = extracted[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = extracted[n + 24]
        freq[new_val] = freq.get(new_val, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
        
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
