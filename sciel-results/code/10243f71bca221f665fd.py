import time
import heapq
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    MIN_N = 10000
    MAX_N = 1000000
    
    # Sieve to get primes up to a limit
    def sieve_primes(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We'll generate numbers in order of increasing LPF using a priority queue
    # Strategy: start with small primes, multiply by small primes to generate candidates
    # Track LPF for each generated number
    
    # Generate primes up to 100 for the multiplication steps
    small_primes = sieve_primes(100)
    
    # Priority queue: (lpf, value)
    # We want to generate numbers whose LPF is small
    # Start with all small primes and 1
    
    # To find consecutive pairs, we need to track which numbers we've seen
    # and check if n and n+1 are both in our "smooth" set
    
    # Alternative approach: Use the priority queue to generate numbers in order of LPF
    # and maintain a set of recently generated numbers to check for consecutiveness
    
    best_n = MIN_N
    best_val = float('inf')
    
    # We'll use a different strategy: 
    # 1. Precompute LPF for all numbers up to MAX_N using a sieve
    # 2. Scan for the best consecutive pair
    
    # Sieve for largest prime factor
    lpf = list(range(MAX_N + 1))
    for i in range(2, isqrt(MAX_N) + 1):
        if lpf[i] == i:  # i is prime
            for j in range(i, MAX_N + 1, i):
                lpf[j] = i
    
    # For numbers that are prime, lpf[n] = n
    # For composite numbers, lpf[n] is the largest prime factor
    
    # Now scan for best consecutive pair
    for n in range(MIN_N, MAX_N):
        if time.time() > deadline - 2:
            break
        
        # LPF of n(n+1) = max(lpf[n], lpf[n+1])
        # But we need the actual largest prime factor, not just the smallest
        # The sieve above gives us the SMALLEST prime factor, not largest
        
        # Let me redo: compute largest prime factor properly
        pass
    
    # Actually, let's use a proper largest prime factor sieve
    # lpf[n] = largest prime factor of n
    
    lpf_arr = [0] * (MAX_N + 1)
    for i in range(2, MAX_N + 1):
        if lpf_arr[i] == 0:  # i is prime
            for j in range(i, MAX_N + 1, i):
                lpf_arr[j] = i  # i is a prime factor, keep updating to get largest
    
    # Now scan
    for n in range(MIN_N, MAX_N):
        if time.time() > deadline - 2:
            break
        
        max_lpf = max(lpf_arr[n], lpf_arr[n+1])
        
        if max_lpf < best_val:
            best_val = max_lpf
            best_n = n
    
    return int(best_n)
